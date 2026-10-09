"""A plan that leaves an analysis the question asks for unanswered is not offered for approval.

The question requirements owner refuses approval with a typed reason: the
plan does not answer an analysis the question asks for
(``question_requirement_not_covered``), it cannot
(``question_requirement_capability_gap``), or the record of what the question
asks cannot be read where the plan is offered for review
(``question_requirements_unreadable``).  The conversation reads each as a
plan under review with that reason as its next action, offers no approval,
lets a fresh candidate plan be generated -- one that reads no patient row
before its review -- and shows the run with nothing analysed.  Each reader
spreads the same registry (``approval_stops.PLAN_APPROVAL_STOPS``), so these
stops reach all of them.  The records of what the question asks, with every
claim the host could not verify marked, are run files beside the review: the
planning record and the judgment of the plan offered for review, which says
whether it judged the plan under review.  Each opens in the conversation with
every field it holds, so a reader sees which requirement stopped the plan and
why.  Synthetic studies only.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.orchestration.progressive_planning import (
    question_requirement_outcome,
)
from easyicu.research_agent.planning.approval_stops import PLAN_APPROVAL_STOPS
from easyicu.research_agent.planning.question_requirements import (
    QUESTION_REQUIREMENT_STOP_CODES,
    QUESTION_REQUIREMENTS_FILENAME,
    QUESTION_REQUIREMENTS_REVIEW_FILENAME,
    QUESTION_REQUIREMENTS_REVIEW_SCHEMA_VERSION,
    QUESTION_REQUIREMENTS_SCHEMA_VERSION,
    analysis_plan_sha256,
    question_requirements_on_plan_under_review,
)
from easyicu.research_agent.schema import AnalysisPlan
from easyicu.webserver import agent_pipeline_runs, agent_runs, run_file_guide
from easyicu.webserver import study_contexts as study_context_owner
from easyicu.webserver.pi_copilot.projections import project_run_row
from easyicu.webserver.pi_copilot.service import PiCopilotService
from easyicu.webserver.pi_copilot.workflow import (
    build_research_workflow_snapshot,
    host_decision_offers,
)
from easyicu.webserver.routes import agent as agent_routes
from tests.research_agent.planning.family_spec_fixtures import (
    _prediction_context,
    _prediction_payload,
    _request,
    _run,
)
from tests.webserver.copilot.pi_copilot_contract_fixtures import FakeGateway
from tests.webserver.copilot.research_workflow_fixtures import complete_study

_NOT_COVERED = "question_requirement_not_covered"
_GAP = "question_requirement_capability_gap"
_UNREADABLE = "question_requirements_unreadable"
_STOPS = [_NOT_COVERED, _GAP, _UNREADABLE]
_RUN = "run-question-requirement-stop"


def _run_and_review(
    reasons: list[str], *, digest: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    run = {
        "run_id": _RUN,
        "run_type": "full",
        "engine": "easyicu.research_agent.pipeline",
        "gate_status": "blocked",
        "run_status": "human_review_pending",
        "pending_review_reason_codes": list(reasons),
        "scientific_configuration_sha256": digest,
        "artifact_names": ["agent_plan.json", "scientific_plan_review.json"],
    }
    review = {
        "run_id": _RUN,
        "resumable_here": True,
        "scientific_configuration_sha256": digest,
        "budget_mode": "full_reviewed",
        "requests": [
            {
                "review_id": f"review-{index}",
                "kind": "scientific_stop",
                "summary": "This plan cannot be approved.",
                "authority_sha256": "b" * 64,
                "reason_code": reason,
                "approval_allowed": False,
            }
            for index, reason in enumerate(reasons)
        ],
        "plan_approval_allowed": False,
        "scientific_plan_review": {
            "status": "ready_for_approval",
            "approval_allowed": True,
            "score": 90,
            "findings": [],
        },
    }
    return run, review


def _workflow(study: Mapping[str, Any], reasons: list[str]):
    digest = study_context_owner.scientific_configuration_sha256(dict(study))
    run, review = _run_and_review(reasons, digest=digest)
    snapshot = build_research_workflow_snapshot(
        study=study,
        active_export_present=True,
        active_job=None,
        latest_run=run,
        plan_review_authority=review,
    )
    offers = host_decision_offers(
        snapshot, study=study, latest_run=run, plan_review_authority=review
    )
    return snapshot, offers


def test_the_registry_holds_the_owners_stops() -> None:
    assert QUESTION_REQUIREMENT_STOP_CODES == tuple(_STOPS)
    assert set(_STOPS) <= set(PLAN_APPROVAL_STOPS)
    # A fresh candidate plan can be generated from each stop.
    assert set(_STOPS) <= agent_routes._CANDIDATE_PLAN_WORKFLOW_CODES


@pytest.mark.parametrize("reason", _STOPS)
def test_the_plan_is_under_review_with_the_reason_as_its_next_action(
    reason: str,
) -> None:
    snapshot, offers = _workflow(complete_study(), [reason])

    stage = next(row for row in snapshot.stages if row.id == "plan")
    assert (stage.status, stage.reason_code) == ("review_required", reason)
    assert snapshot.next_action_code == reason
    assert snapshot.plan_execution_ready is False
    assert offers.plan_review is None


def test_a_population_stop_is_named_before_a_question_stop() -> None:
    snapshot, _offers = _workflow(
        complete_study(), [_GAP, "population_inclusion_requires_extraction"]
    )

    assert snapshot.next_action_code == "population_inclusion_requires_extraction"


class _Request:
    def __init__(self, reason: str) -> None:
        self.payload = {"reason": reason, "approval_allowed": False}


@pytest.mark.parametrize("reason", _STOPS)
def test_the_pending_review_keeps_the_typed_reason(reason: str) -> None:
    current = {
        "schema_version": agent_pipeline_runs.CURRENT_SCIENTIFIC_REVIEW_SCHEMA_VERSION,
        "approval_allowed": True,
    }

    assert (
        agent_pipeline_runs._pending_review_reason_code(
            request=_Request(reason),
            plan_recommendation_complete=True,
            scientific_plan_review=current,
        )
        == reason
    )


@pytest.mark.parametrize("reason", _STOPS)
def test_the_conversation_reads_the_run_with_nothing_analysed(reason: str) -> None:
    run, _review = _run_and_review([reason], digest="e" * 64)
    run["artifact_names"] += ["result_tables.json", "manuscript_draft.json"]

    projected = project_run_row(run)

    assert projected["execution_phase"] == "plan_review"
    assert projected["plan_approval_allowed"] is False
    assert projected["analysis_executed"] is False
    assert (
        projected["artifact_semantics"]
        == "plan_stage_placeholders_not_analysis_results"
    )


@pytest.mark.parametrize("mode", ["fresh", "auto"])
@pytest.mark.parametrize("reason", _STOPS)
def test_a_fresh_plan_after_the_stop_reads_no_patient_row(
    monkeypatch: pytest.MonkeyPatch, reason: str, mode: str
) -> None:
    monkeypatch.setattr(
        agent_routes,
        "build_project_workflow_projection",
        lambda **_kwargs: SimpleNamespace(
            workflow=SimpleNamespace(next_action_code=reason)
        ),
    )

    # The host keeps the regenerated plan a candidate: metadata only.
    assert agent_routes._candidate_plan_only_authorized(
        {"study_context_id": "study-1", "planner_start_mode": mode}
    )


def _plan() -> Any:
    context = _prediction_context()
    request = _request(context, cohort_mode=None)
    _llm, result = _run(
        context,
        [json.dumps(_prediction_payload(request, features=["age", "sex", "hr_max"]))],
        required_primary_cohort_selection_mode=None,
    )
    return result.output


def _records(tmp_path: Path, plan: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    planning = {
        "schema_version": QUESTION_REQUIREMENTS_SCHEMA_VERSION,
        "route": "family_template",
        "compiled_plan_sha256": "c" * 64,
    }
    review = {
        "schema_version": QUESTION_REQUIREMENTS_REVIEW_SCHEMA_VERSION,
        "plan_sha256": analysis_plan_sha256(plan),
        "judged": [
            {
                "id": "r1",
                "disposition": "attested",
                "reason_code": "question_requirement_attested",
                "verified_by_host": False,
            }
        ],
    }
    for name, record in (
        (QUESTION_REQUIREMENTS_FILENAME, planning),
        (QUESTION_REQUIREMENTS_REVIEW_FILENAME, review),
    ):
        (tmp_path / name).write_text(json.dumps(record), encoding="utf-8")
    return planning, review


def test_the_records_of_what_the_question_asks_are_run_files(tmp_path: Path) -> None:
    plan = _plan()
    planning, review = _records(tmp_path, plan)

    projected = agent_pipeline_runs._load_question_requirements(
        tmp_path, plan.model_dump(mode="json")
    )

    assert projected == {
        QUESTION_REQUIREMENTS_FILENAME: planning,
        QUESTION_REQUIREMENTS_REVIEW_FILENAME: {
            **review,
            "judged_on_plan_under_review": True,
        },
    }
    for name in (QUESTION_REQUIREMENTS_FILENAME, QUESTION_REQUIREMENTS_REVIEW_FILENAME):
        assert name in agent_runs._RUN_ARTIFACT_NAMES
        assert name in run_file_guide._ORDER


def test_the_judgment_says_when_it_judged_another_plan(tmp_path: Path) -> None:
    plan = _plan()
    _planning, review = _records(tmp_path, plan)
    primary = next(
        item for item in plan.steps if item.planned_analysis_role == "primary"
    )
    reread = plan.model_copy(
        update={
            "steps": [
                item.model_copy(update={"inputs": [*item.inputs, "map_min"]})
                if item is primary
                else item
                for item in plan.steps
            ]
        }
    ).model_dump(mode="json")

    projected = agent_pipeline_runs._load_question_requirements(tmp_path, reread)

    assert (
        projected[QUESTION_REQUIREMENTS_REVIEW_FILENAME]["judged_on_plan_under_review"]
        is False
    )
    # Another record, or none, is not projected.
    (tmp_path / QUESTION_REQUIREMENTS_REVIEW_FILENAME).write_text(
        json.dumps({**review, "schema_version": "other/1"}), encoding="utf-8"
    )
    assert QUESTION_REQUIREMENTS_REVIEW_FILENAME not in (
        agent_pipeline_runs._load_question_requirements(tmp_path, reread)
    )
    assert (
        agent_pipeline_runs._load_question_requirements(tmp_path / "absent", reread)
        == {}
    )


def test_the_judgment_binds_the_plan_its_review_authority_binds(
    tmp_path: Path,
) -> None:
    payload = _plan().model_dump(mode="json")
    # A cohort concept the host sealed for this plan: registered only in the
    # scope that plan was validated in, never in this process.
    payload["cohort"]["inclusion"][0]["concept_id"] = "a_concept_sealed_for_this_plan"
    with pytest.raises(ValueError):
        AnalysisPlan.model_validate(payload)
    (tmp_path / QUESTION_REQUIREMENTS_REVIEW_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": QUESTION_REQUIREMENTS_REVIEW_SCHEMA_VERSION,
                "plan_sha256": canonical_sha256(payload),
            }
        ),
        encoding="utf-8",
    )

    projected = agent_pipeline_runs._load_question_requirements(tmp_path, payload)

    assert (
        projected[QUESTION_REQUIREMENTS_REVIEW_FILENAME]["judged_on_plan_under_review"]
        is True
    )


#: One analysis the plan answers and one estimand it declares it cannot.
_ASKED = [
    {
        "id": "r1",
        "kind": "analysis",
        "quote": "predict in-hospital mortality",
        "concepts": ["lactate_max"],
        "coverage": "plan",
        "gap": None,
        "note": None,
    },
    {
        "id": "r2",
        "kind": "estimand",
        "quote": "discrimination and calibration",
        "concepts": [],
        "coverage": "capability_gap",
        "gap": {
            "requirement": "estimand_unsupported",
            "concept": None,
            "element": "analysis",
            "detail": "This plan cannot estimate the named measure on held-out rows.",
        },
        "note": None,
    },
]


def _written_records(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Both records as the planning owner writes them for a synthetic study."""

    base = _prediction_context()
    constraints = json.loads(base.user_preferences.data_constraints or "{}")
    constraints["question_named_concepts"] = [
        {"concepts": ["lactate_max"], "evidence": "labs"}
    ]
    context = base.model_copy(
        update={
            # The question names both requirements the plan is judged on.
            "research_question": (
                "Among adult ICU stays, how well do first-24-hour vitals, labs, "
                "and demographics predict in-hospital mortality, in "
                "discrimination and calibration?"
            ),
            "user_preferences": base.user_preferences.model_copy(
                update={"data_constraints": json.dumps(constraints)}
            ),
        }
    )
    request = _request(context, cohort_mode=None)
    payload = {
        **_prediction_payload(
            request, features=["age", "sex", "hr_max", "lactate_max"]
        ),
        "question_requirements": _ASKED,
    }
    _llm, result = _run(
        context, [json.dumps(payload)], required_primary_cohort_selection_mode=None
    )
    findings = question_requirement_outcome(
        context=context, plan=result.output, facts=result.facts, run_dir=run_dir
    )
    question_requirements_on_plan_under_review(
        findings, plan=result.output, run_dir=run_dir
    )
    return {
        name: json.loads((run_dir / name).read_text(encoding="utf-8"))
        for name in (
            QUESTION_REQUIREMENTS_FILENAME,
            QUESTION_REQUIREMENTS_REVIEW_FILENAME,
        )
    }


@pytest.mark.parametrize(
    "name", [QUESTION_REQUIREMENTS_FILENAME, QUESTION_REQUIREMENTS_REVIEW_FILENAME]
)
def test_each_record_opens_in_the_conversation_with_every_field_it_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    record = _written_records(run_dir)[name]
    # What a reader needs to see why the plan stopped is in the record.
    (gap,) = [row for row in record["judged"] if row["disposition"] == "capability_gap"]
    assert (gap["quote"], gap["gap"]["detail"]) == (
        _ASKED[1]["quote"],
        _ASKED[1]["gap"]["detail"],
    )
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json", gateway=FakeGateway()
    )
    service.project_store.bind("project-q", "study-q")
    monkeypatch.setattr(
        agent_runs,
        "list_run_history",
        lambda *, study_id, **_kwargs: {
            "runs": (
                [{"run_id": _RUN, "project_dir": str(run_dir)}]
                if study_id == "study-q"
                else []
            )
        },
    )
    monkeypatch.setattr(
        agent_runs,
        "read_run_review",
        lambda project_dir: {
            "ok": True,
            "gate": {"status": "blocked"},
            "readiness": {
                "status": "blocked",
                "signed": False,
                "signoff_stale": False,
                "reportable": False,
            },
        },
    )

    opened = service.get_research_artifact(
        project_id="project-q", run_id=_RUN, artifact_name=name
    )

    # The run file projection withholds no field and the browser projection
    # drops none: the conversation shows the record as written.
    assert opened["payload"] == record
