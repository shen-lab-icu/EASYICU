"""A plan under review shows what the question asks of it, each line with its source.

The planning owner records each analysis the question asks for and how the
host judged a plan on it, twice: on the plan the Planner compiled
(``question_requirements.json``) and on the plan a review request offers
(``question_requirements_review.json``), which an approval rests on.  Only the
approval stops reached the card; the plan's own claims, which the host could
not verify, and the concepts a route without requirements left unstated were
in run files nobody opened.  The review summary now carries the card's view
(``pi_copilot.question_requirement_notes``): the judgment of the plan under
review, or, when there is none or it judged another plan, the planning record,
with a line saying the judgments may not hold.  The card lists what a reviewer
must look at: a requirement the plan does not answer or cannot carry out, a
claim the plan made that EasyICU did not verify, and each concept the question
names on a route that lists none.  Verified answers are counted, and the card
links the record it read.  A record that exists but cannot be read says so.

Synthetic records and plans; no benchmark item.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.orchestration.workflow import (
    HumanReviewPending,
    HumanReviewRequest,
)
from easyicu.research_agent.planning.question_requirements import (
    QUESTION_REQUIREMENTS_FILENAME,
    QUESTION_REQUIREMENTS_REVIEW_FILENAME,
    QUESTION_REQUIREMENTS_REVIEW_SCHEMA_VERSION,
    QUESTION_REQUIREMENTS_SCHEMA_VERSION,
    JudgedRequirement,
    QuestionRequirement,
    analysis_plan_sha256,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver import study_contexts as study_context_owner
from easyicu.webserver.pi_copilot.question_requirement_notes import (
    UNREADABLE_REASON,
    project_question_requirement_notes,
)
from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot
from easyicu.webserver.routes import agent as agent_routes
from tests.support.node import run_node
from tests.webserver.copilot.research_workflow_fixtures import complete_study

STATIC = Path(agent_routes.__file__).resolve().parents[1] / "static"
PLANNING = QUESTION_REQUIREMENTS_FILENAME
REVIEW = QUESTION_REQUIREMENTS_REVIEW_FILENAME


def _row(rid: str, disposition: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": rid,
        "kind": "analysis",
        "quote": f"the {rid} analysis",
        "concepts": [],
        "disposition": disposition,
        "verified_by_host": disposition in {"covered", "not_covered"},
        "gap": None,
        "gap_verification": None,
        "note": None,
        **extra,
    }


def _review(**extra: Any) -> dict[str, Any]:
    """The judgment of the plan under review, as the run projects it."""

    return {
        "schema_version": QUESTION_REQUIREMENTS_REVIEW_SCHEMA_VERSION,
        "plan_sha256": "c" * 64,
        "route": "family_template",
        "judged_on_plan_under_review": True,
        "judged": [
            _row("r1", "covered"),
            _row(
                "r2", "definition_only", kind="definition", note="the comparison score"
            ),
            _row("r3", "attested", kind="estimand"),
            _row("r4", "covered"),
            _row(
                "r5",
                "capability_gap",
                kind="benchmark",
                gap={"requirement": "design_element_unsupported"},
                gap_verification="unverifiable",
            ),
            _row("r6", "not_covered", kind="subgroup"),
        ],
        "unstated": [],
        **extra,
    }


def _planning(**extra: Any) -> dict[str, Any]:
    """The planning record, judged on the compiled plan."""

    record = _review(**extra)
    for key in ("plan_sha256", "judged_on_plan_under_review"):
        record.pop(key, None)
    return {**record, "schema_version": QUESTION_REQUIREMENTS_SCHEMA_VERSION}


def _notes(review: Mapping[str, Any] | None = None, **records: Any) -> dict[str, Any]:
    """The card's view of the review judgment, and of any planning record."""

    projected = {REVIEW: review if review is not None else _review(), **records}
    return project_question_requirement_notes(projected, recorded=list(projected))


# --- the host's projection --------------------------------------------------------


def test_a_run_without_a_record_projects_nothing() -> None:
    assert project_question_requirement_notes(None, recorded=()) is None
    assert project_question_requirement_notes({}, recorded=[]) is None


def test_a_record_that_cannot_be_read_is_reported_not_dropped() -> None:
    for written in ([PLANNING], [REVIEW], [PLANNING, REVIEW]):
        notes = project_question_requirement_notes({}, recorded=written)

        assert notes == {
            "status": "unavailable",
            "reason_code": UNREADABLE_REASON,
            "items": [],
            "covered_count": 0,
            "unstated": [],
        }
    # The display's own code: the planning owner's stop of a similar name says
    # the plan was not judged, which is a different fact.
    assert UNREADABLE_REASON == "question_requirements_record_unreadable"


def test_the_card_reads_the_judgment_of_the_plan_under_review() -> None:
    planning = _planning(judged=[_row("p1", "not_covered")])

    notes = _notes(**{PLANNING: planning})

    assert (notes["record"], notes["judged_on_plan_under_review"]) == (REVIEW, True)
    assert [item["id"] for item in notes["items"]] == ["r6", "r5", "r3", "r2"]


@pytest.mark.parametrize("review_kept", [False, True])
def test_otherwise_it_reads_the_planning_record_which_may_not_hold(
    review_kept: bool,
) -> None:
    planning = _planning(judged=[_row("p1", "not_covered")])
    records: dict[str, Any] = {PLANNING: planning}
    if review_kept:
        # A judgment of another plan: the plan changed after it was judged.
        records[REVIEW] = _review(judged_on_plan_under_review=False)

    notes = project_question_requirement_notes(records, recorded=list(records))

    assert (notes["record"], notes["judged_on_plan_under_review"]) == (PLANNING, False)
    assert [item["id"] for item in notes["items"]] == ["p1"]


def test_a_judgment_of_another_plan_is_shown_when_nothing_else_is() -> None:
    notes = _notes(_review(judged_on_plan_under_review=False))

    assert (notes["record"], notes["judged_on_plan_under_review"]) == (REVIEW, False)


def test_each_item_names_its_source_and_verified_answers_are_counted() -> None:
    notes = _notes()

    assert notes is not None
    assert (notes["status"], notes["covered_count"]) == ("shown", 2)
    assert [
        (item["id"], item["disposition"], item["source"]) for item in notes["items"]
    ] == [
        ("r6", "not_covered", "host_verified"),
        ("r5", "capability_gap", "plan_declared"),
        ("r3", "attested", "plan_declared"),
        ("r2", "definition_only", "plan_declared"),
    ]
    gap = notes["items"][1]
    assert (gap["gap_requirement"], gap["gap_verification"]) == (
        "design_element_unsupported",
        "unverifiable",
    )
    assert notes["items"][3]["note"] == "the comparison score"


def test_a_gap_the_host_verified_is_the_hosts() -> None:
    record = _review(
        judged=[
            _row(
                "r1",
                "capability_gap",
                gap={"requirement": "design_element_unsupported"},
                gap_verification="verified",
                verified_by_host=True,
            )
        ]
    )

    (item,) = _notes(record)["items"]

    assert item["source"] == "host_verified"


@pytest.mark.parametrize(
    ("disposition", "verification"),
    [
        ("covered", None),
        ("not_covered", None),
        ("capability_gap", "verified"),
        ("capability_gap", "unverified"),
        ("capability_gap", "unverifiable"),
        ("attested", None),
        ("definition_only", None),
    ],
)
def test_an_items_source_is_the_planning_owners_verdict(
    disposition: str, verification: str | None
) -> None:
    # The row as the planning owner writes it: the card does not decide anew
    # what the host verified.
    judged = JudgedRequirement(
        requirement=QuestionRequirement(
            id="r1", kind="analysis", quote="the r1 analysis", coverage="plan"
        ),
        disposition=disposition,
        reason_code="question_requirement_judged",
        verification=verification,
    )

    notes = _notes(_review(judged=[judged.row()]))

    if disposition == "covered":
        assert (notes["items"], notes["covered_count"]) == ([], 1)
    else:
        assert [item["source"] for item in notes["items"]] == [
            "host_verified" if judged.verified_by_host else "plan_declared"
        ]


def test_concepts_a_route_without_requirements_left_unstated_say_whether_read() -> None:
    record = _review(
        route="outline",
        judged=[],
        unstated=[
            {
                "concepts": ["apache_iva"],
                "evidence": "APACHE IVa",
                "reading_step_ids": ["primary"],
                "cohort_criterion": False,
            },
            {
                "concepts": ["sofa"],
                "evidence": "SOFA",
                "reading_step_ids": [],
                "cohort_criterion": True,
            },
            {
                "concepts": ["lactate"],
                "evidence": "lactate",
                "reading_step_ids": [],
                "cohort_criterion": False,
            },
        ],
    )

    notes = _notes(record)

    assert [(row["evidence"], row["read"]) for row in notes["unstated"]] == [
        ("APACHE IVa", True),
        ("SOFA", True),
        ("lactate", False),
    ]


def test_question_and_model_text_is_bounded() -> None:
    record = _review(
        judged=[
            _row("r1", "attested", quote="x " * 400),
            _row("r2", "definition_only", note="  long\n note " * 200),
        ]
    )

    notes = _notes(record)

    assert len(notes["items"][0]["quote"]) == 240
    assert len(notes["items"][1]["note"]) == 300
    assert "\n" not in notes["items"][1]["note"]


# --- the paused run hands the record to the review --------------------------------


def _pending(run_dir: Path) -> tuple[HumanReviewPending, AnalysisPlan]:
    plan = AnalysisPlan(
        research_question="Does the model predict mortality?",
        steps=[
            AnalysisStep(
                step_id="primary",
                intent="Fit the model",
                method="descriptive",
                inputs=[],
                expected_outputs=["table:model"],
            )
        ],
    )
    authority = PlanReviewAuthority.create(plan=plan)
    request = HumanReviewRequest.create(
        kind="scientific_stop",
        summary="Review the digest-bound plan before analysis.",
        authority_sha256="a" * 64,
        payload={
            "reason": "operator_plan_approval_required",
            "plan_review_authority": authority.model_dump(mode="json"),
        },
    )
    pending = HumanReviewPending(
        run_id="run-requirements",
        thread_id="thread-requirements",
        run_dir=str(run_dir),
        requests=(request,),
    )
    return pending, plan


def test_the_paused_run_passes_its_records_and_says_which_it_wrote(
    tmp_path: Path,
) -> None:
    pending, plan = _pending(tmp_path)

    assert agent_pipeline_runs._pending_question_requirements(tmp_path, pending) == {
        "question_requirements": {},
        "question_requirements_recorded": [],
    }

    (tmp_path / PLANNING).write_text(json.dumps(_planning()), encoding="utf-8")
    review = _review(plan_sha256=analysis_plan_sha256(plan))
    review.pop("judged_on_plan_under_review")
    (tmp_path / REVIEW).write_text(json.dumps(review), encoding="utf-8")
    passed = agent_pipeline_runs._pending_question_requirements(tmp_path, pending)
    assert passed["question_requirements_recorded"] == [PLANNING, REVIEW]
    assert set(passed["question_requirements"]) == {PLANNING, REVIEW}
    assert passed["question_requirements"][REVIEW]["judged_on_plan_under_review"] is True

    (tmp_path / REVIEW).write_text(
        json.dumps({**review, "plan_sha256": "d" * 64}), encoding="utf-8"
    )
    passed = agent_pipeline_runs._pending_question_requirements(tmp_path, pending)
    assert passed["question_requirements"][REVIEW]["judged_on_plan_under_review"] is False

    (tmp_path / PLANNING).write_text("{not json", encoding="utf-8")
    (tmp_path / REVIEW).write_text("{not json", encoding="utf-8")
    assert agent_pipeline_runs._pending_question_requirements(tmp_path, pending) == {
        "question_requirements": {},
        "question_requirements_recorded": [PLANNING, REVIEW],
    }


def test_the_paused_runs_review_carries_both_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pending, plan = _pending(tmp_path)
    (tmp_path / PLANNING).write_text(json.dumps(_planning()), encoding="utf-8")
    review = _review(plan_sha256=analysis_plan_sha256(plan))
    review.pop("judged_on_plan_under_review")
    (tmp_path / REVIEW).write_text(json.dumps(review), encoding="utf-8")
    registry = agent_pipeline_runs.PendingReviewRegistry()
    monkeypatch.setattr(agent_pipeline_runs, "_PENDING_REVIEWS", registry)
    registry.register(
        agent_pipeline_runs._PendingRun(
            pipeline=SimpleNamespace(),
            pending=pending,
            wrapper_dir=tmp_path / "wrapper",
            study=complete_study(),
            provider={},
            acquisition=SimpleNamespace(),
            created_at=1.0,
        )
    )

    projected = agent_pipeline_runs.pending_review(pending.run_id)

    assert projected is not None
    assert projected["question_requirements_recorded"] == [PLANNING, REVIEW]
    assert set(projected["question_requirements"]) == {PLANNING, REVIEW}
    assert projected["question_requirements"][REVIEW]["judged_on_plan_under_review"] is True


def _snapshot(review_extra: Mapping[str, Any]):
    study = complete_study()
    digest = study_context_owner.scientific_configuration_sha256(dict(study))
    run = {
        "run_id": "run-requirements",
        "run_type": "full",
        "engine": "easyicu.research_agent.pipeline",
        "gate_status": "blocked",
        "run_status": "human_review_pending",
        "pending_review_reason_codes": ["operator_plan_approval_required"],
        "scientific_configuration_sha256": digest,
        "artifact_names": ["agent_plan.json", "scientific_plan_review.json"],
    }
    review = {
        "run_id": "run-requirements",
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


def test_the_review_summary_carries_the_cards_view_of_the_record() -> None:
    snapshot = _snapshot(
        {
            "question_requirements": {REVIEW: _review()},
            "question_requirements_recorded": [REVIEW],
        }
    )

    notes = snapshot.plan_review_summary["question_requirements"]
    assert (notes["status"], notes["record"]) == ("shown", REVIEW)
    assert [item["id"] for item in notes["items"]] == ["r6", "r5", "r3", "r2"]

    unread = _snapshot(
        {"question_requirements": {}, "question_requirements_recorded": [REVIEW]}
    )
    assert (
        unread.plan_review_summary["question_requirements"]["status"] == "unavailable"
    )
    # A run that wrote no record adds nothing to the summary.
    assert "question_requirements" not in _snapshot({}).plan_review_summary


# --- the card ---------------------------------------------------------------------


def _node() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    return node


def _notes_html(notes: Mapping[str, Any] | None, *, link: bool = False) -> str:
    script = r"""
global.window = {};
require(process.argv[1]);
let api = null;
window.EasyICU = { guidedPi: { declare: (_name, value) => { api = value; } } };
require(process.argv[2]);
const summary = JSON.parse(process.argv[3]);
const link = process.argv[4] === 'link';
process.stdout.write(api.notesHtml(summary, {
  tr: (en, zh) => zh,
  esc: window.EU_HTML.esc,
  recordLink: artifact => (link ? `<a data-resource="${artifact}"></a>` : ''),
}));
"""
    result = run_node(
        _node(),
        script,
        str((STATIC / "js" / "html-escape.js").resolve()),
        str((STATIC / "js" / "screens-guided-pi-question-requirements.js").resolve()),
        json.dumps({"question_requirements": notes}),
        "link" if link else "",
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout


def _lines(html: str) -> list[str]:
    return [part.split("</li>", 1)[0] for part in html.split("<li>")[1:]]


def test_the_card_names_each_requirement_and_its_source() -> None:
    notes = _notes(
        _review(
            judged=[
                *_review()["judged"],
                _row(
                    "r7",
                    "capability_gap",
                    gap={"requirement": "design_element_unsupported"},
                    gap_verification="verified",
                    verified_by_host=True,
                ),
            ]
        )
    )

    html = _notes_html(notes, link=True)

    assert "<strong>题面要求</strong>" in html
    assert _lines(html) == [
        "这份计划没有回答：「the r6 analysis」",
        "规划声明这份计划做不到（EasyICU 未核实）：「the r5 analysis」",
        "这份计划做不到：「the r7 analysis」",
        "计划声明，EasyICU 未核实：「the r3 analysis」",
        "计划声明，EasyICU 未核实：「the r2 analysis」（计划说它只用于定义：the comparison score）",
        "另有 2 项已由计划步骤回答（EasyICU 已核实）",
    ]
    assert html.count("data-resource=") == 1
    assert f'data-resource="{REVIEW}"' in html


def test_unstated_concepts_ask_the_reviewer_what_to_check() -> None:
    notes = _notes(
        _review(
            route="outline",
            judged=[],
            unstated=[
                {
                    "concepts": ["apache_iva"],
                    "evidence": "APACHE IVa",
                    "reading_step_ids": ["primary"],
                },
                {
                    "concepts": ["lactate"],
                    "evidence": "lactate",
                    "reading_step_ids": [],
                },
            ],
        )
    )

    assert _lines(_notes_html(notes)) == [
        "规划时没有逐项列出题面要求；计划读取了题面提到的「APACHE IVa」，请核对计划是否回答了它",
        "规划时没有逐项列出题面要求；题面提到的「lactate」没有被任何步骤读取，请核对计划是否遗漏了它",
    ]


def test_the_planning_record_says_its_judgments_may_not_hold() -> None:
    notes = project_question_requirement_notes(
        {PLANNING: _planning(judged=[_row("p1", "not_covered")])}, recorded=[PLANNING]
    )

    html = _notes_html(notes, link=True)

    assert _lines(html) == [
        "这份计划没有回答：「the p1 analysis」",
        "这些判定针对改动前的计划，对当前计划可能已不成立",
    ]
    assert f'data-resource="{PLANNING}"' in html


def test_question_text_is_escaped_and_clipped_on_the_card() -> None:
    notes = {
        "status": "shown",
        "items": [
            {
                "disposition": "attested",
                "source": "plan_declared",
                "quote": "<img src=x onerror=alert(1)>" + "y" * 400,
            },
        ],
        "unstated": [],
        "covered_count": 0,
        "judged_on_plan_under_review": True,
    }

    html = _notes_html(notes)

    assert "<img" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    (line,) = _lines(html)
    assert line.endswith("…」")
    assert len(line) < 320


def test_an_unreadable_record_is_one_line_with_nothing_to_open() -> None:
    html = _notes_html(
        project_question_requirement_notes({}, recorded=[REVIEW]), link=True
    )

    assert _lines(html) == ["题面要求记录无法读取，未显示"]
    assert "data-resource" not in html


def test_a_record_with_only_verified_answers_adds_nothing_to_the_card() -> None:
    notes = _notes(_review(judged=[_row("r1", "covered")]))

    assert _notes_html(notes) == ""
    assert _notes_html(None) == ""


def _card_html(notes: Mapping[str, Any], code: str) -> str:
    script = r"""
global.window = { EU_LANG: 'zh' };
require(process.argv[1]);
let notesApi = null;
let declared = null;
window.EasyICU = { guidedPi: {
  declare: (name, api) => { if (name === 'questionRequirements') notesApi = api; else declared = api; },
  optional: name => (name === 'questionRequirements' ? notesApi : null),
} };
require(process.argv[2]);
require(process.argv[3]);
const summary = { run_id: 'run-requirements', question_requirements: JSON.parse(process.argv[4]) };
const owner = declared.create({
  tr: (en, zh) => zh || en,
  esc: window.EU_HTML.esc,
  iconHtml: () => '',
  resourceButton: resource => `<a data-resource="${resource.artifact}"></a>`,
  sessionIsStale: () => false,
  busy: () => false,
  session: () => ({ binding: { run_id: 'run-requirements' } }),
  workflow: () => ({ next_action_code: process.argv[5], plan_review_summary: summary }),
});
process.stdout.write(owner.workflowConfirmationHtml());
"""
    result = run_node(
        _node(),
        script,
        str((STATIC / "js" / "html-escape.js").resolve()),
        str((STATIC / "js" / "screens-guided-pi-question-requirements.js").resolve()),
        str((STATIC / "js" / "screens-guided-pi-confirmation.js").resolve()),
        json.dumps(notes),
        code,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout


@pytest.mark.parametrize(
    "code", ["question_requirement_not_covered", "question_requirement_capability_gap"]
)
def test_a_stop_card_shows_the_block_and_links_the_record_once(code: str) -> None:
    html = _card_html(_notes(), code)

    assert 'class="gpi-question-requirements"' in html
    assert "这份计划没有回答：「the r6 analysis」" in html
    # The stop card links the judgment already, so the block adds no second link.
    assert html.count(f'data-resource="{REVIEW}"') == 1


def test_the_approval_card_shows_the_block_with_its_own_link() -> None:
    html = _card_html(_notes(), "operator_plan_approval_required")

    assert 'class="gpi-question-requirements"' in html
    assert "计划声明，EasyICU 未核实：「the r3 analysis」" in html
    assert html.count(f'data-resource="{REVIEW}"') == 1


def test_a_stop_card_reading_the_planning_record_links_it_beside_the_judgment() -> None:
    notes = project_question_requirement_notes(
        {PLANNING: _planning()}, recorded=[PLANNING]
    )

    html = _card_html(notes, "question_requirement_not_covered")

    assert html.count(f'data-resource="{PLANNING}"') == 1
    assert "这些判定针对改动前的计划，对当前计划可能已不成立" in html


def test_the_blocks_record_link_is_a_full_size_control() -> None:
    # A text link inside the block keeps the card's control height
    # (tools/audit_web_fit.py MIN_HIT, 24 px), as the card's own links do.
    css = (STATIC / "css" / "guided-pi-question-requirements.css").read_text(
        encoding="utf-8"
    )
    rule = re.search(
        r"\.gpi-question-requirements-link \.gpi-resource-link\{[^}]*min-height:(\d+)px",
        css,
    )

    assert rule is not None and int(rule.group(1)) >= 24
