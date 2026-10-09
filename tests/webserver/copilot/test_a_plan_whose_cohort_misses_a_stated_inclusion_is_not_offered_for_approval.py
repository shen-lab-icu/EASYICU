"""A plan whose cohort misses an inclusion the study states is not offered for approval.

Step 2b of the population spec design: the host compiles the plan's cohort
from the population the Planner states, and an inclusion the cohort does not
apply no longer stops planning.  The plan is reviewed, but its approval is
refused with a typed reason: ``population_inclusion_requires_extraction``
when an extraction of the study's own population can apply it, and
``population_inclusion_not_applied`` when nothing can as stated.

The workflow read neither reason as a plan review, so it showed such a plan
as ready and named preparing its data as the next action.  It now shows the
plan under review with the reason as its next action, offers no approval,
and lets a fresh candidate plan be generated; once the study binds the
export extracted for its own population, the plan is superseded and planned
again on that export.

The run the conversation reads is a plan under review too, with nothing
analysed.  Each consumer keeps its own set of the codes that mean a plan
review, and a new code can miss one of them: every such set names the
stops that refuse approval, or this file says why it need not.
Synthetic studies only.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterator, Mapping

import pytest

import easyicu.webserver as webserver_package
from easyicu.research_agent.planning.population_compile import (
    POPULATION_APPROVAL_STOPS,
)
from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver import study_contexts as study_context_owner
from easyicu.webserver.pi_copilot.projections import project_run_row
from easyicu.webserver.pi_copilot.workflow import (
    build_research_workflow_snapshot,
    host_decision_offers,
)
from easyicu.webserver.routes import agent as agent_routes
from tests.webserver.copilot.research_workflow_fixtures import complete_study

_EXTRACT = "population_inclusion_requires_extraction"
_UNAPPLIED = "population_inclusion_not_applied"
_RUN = "run-population-stop"


def _run_and_review(
    study: Mapping[str, Any], reasons: list[str], *, digest: str
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
                "authority_sha256": "a" * 64,
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


def _workflow(study: Mapping[str, Any], reasons: list[str], *, digest: str = ""):
    digest = digest or study_context_owner.scientific_configuration_sha256(dict(study))
    run, review = _run_and_review(study, reasons, digest=digest)
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


def _plan_stage(snapshot) -> tuple[str, str]:
    stage = next(row for row in snapshot.stages if row.id == "plan")
    return stage.status, stage.reason_code


def test_an_inclusion_an_extraction_can_apply_names_that_extraction() -> None:
    snapshot, offers = _workflow(complete_study(), [_EXTRACT])

    assert _plan_stage(snapshot) == ("review_required", _EXTRACT)
    assert snapshot.next_action_code == _EXTRACT
    assert snapshot.plan_execution_ready is False
    assert offers.plan_review is None


def test_an_inclusion_nothing_can_apply_leaves_the_plan_unapprovable() -> None:
    snapshot, offers = _workflow(complete_study(), [_UNAPPLIED])

    assert _plan_stage(snapshot) == ("review_required", _UNAPPLIED)
    assert snapshot.next_action_code == _UNAPPLIED
    assert snapshot.plan_execution_ready is False
    assert offers.plan_review is None
    # The analysis waits for a plan that can be approved.
    analysis = next(row for row in snapshot.stages if row.id == "analysis")
    assert analysis.status == "blocked"


def test_the_extraction_is_named_first_when_both_stop_the_plan() -> None:
    snapshot, _offers = _workflow(complete_study(), [_UNAPPLIED, _EXTRACT])

    assert snapshot.next_action_code == _EXTRACT


def test_the_export_extracted_for_the_study_supersedes_the_plan() -> None:
    study = complete_study()
    planned_on = study_context_owner.scientific_configuration_sha256(dict(study))
    # The study binds the export extracted for its own population (BX).
    rebound = {
        **study,
        "data_source": {
            "path": "/private/prepared/own-population",
            "database": "mimiciv",
        },
    }

    snapshot, _offers = _workflow(rebound, [_EXTRACT], digest=planned_on)

    assert _plan_stage(snapshot) == ("ready", "plan_configuration_superseded")
    assert snapshot.next_action_code in agent_routes._CANDIDATE_PLAN_WORKFLOW_CODES


def test_a_fresh_candidate_plan_from_either_stop_reads_no_patient_row() -> None:
    assert set(POPULATION_APPROVAL_STOPS.values()) == {_EXTRACT, _UNAPPLIED}
    assert {_EXTRACT, _UNAPPLIED} <= agent_routes._CANDIDATE_PLAN_WORKFLOW_CODES


class _Request:
    def __init__(self, reason: str) -> None:
        self.payload = {"reason": reason, "approval_allowed": False}


def test_the_pending_review_keeps_the_typed_reason() -> None:
    current = {
        "schema_version": agent_pipeline_runs.CURRENT_SCIENTIFIC_REVIEW_SCHEMA_VERSION,
        "approval_allowed": True,
    }

    for reason in (_EXTRACT, _UNAPPLIED):
        assert (
            agent_pipeline_runs._pending_review_reason_code(
                request=_Request(reason),
                plan_recommendation_complete=True,
                scientific_plan_review=current,
            )
            == reason
        )


@pytest.mark.parametrize("reason", [_EXTRACT, _UNAPPLIED])
def test_the_conversation_reads_a_plan_under_review_with_nothing_analysed(
    reason: str,
) -> None:
    run, _review = _run_and_review(complete_study(), [reason], digest="d" * 64)
    # The plan stage writes placeholders for the results and the manuscript.
    run["artifact_names"] += ["result_tables.json", "manuscript_draft.json"]

    projected = project_run_row(run)

    assert projected["execution_phase"] == "plan_review"
    assert projected["human_plan_review_pending"] is True
    assert projected["plan_approval_allowed"] is False
    assert projected["analysis_executed"] is False
    assert projected["scientific_results_available"] is False
    assert (
        projected["artifact_semantics"]
        == "plan_stage_placeholders_not_analysis_results"
    )


# A set naming either of these codes reads a plan review.
_PLAN_REVIEW_CODES = frozenset(
    {"operator_plan_approval_required", "plan_scientific_changes_required"}
)
_STOPS_SPREAD = "POPULATION_APPROVAL_STOPS.values()"

# Sets that read a plan review without naming the stops, and why they need not.
_NEED_NOT_NAME_THE_STOPS = {
    ("pi_copilot/tools.py", "_request_replan", "review_declared"): (
        "it picks a review whose live authority may still resume; a plan "
        "that refuses approval never does, and without the code the request "
        "already starts a fresh plan"
    ),
    (
        "pi_copilot/run_authority.py",
        "workflow_authoritative_run",
        "candidate_waits_for_execution_upgrade",
    ): (
        "it keeps a candidate waiting for its execution upgrade over a failed "
        "preparation launched from it; a plan that refuses approval launches none"
    ),
    ("pi_copilot/workflow.py", "_enrich_plan_review", ""): (
        "it sends a reviewer's runtime gap to the host compiler; a population "
        "stop is no such gap, and replacing its next action would hide it"
    ),
}


def _plan_review_code_sets() -> Iterator[tuple[tuple[str, str, str], bool]]:
    """Yield (file, function, assigned name) of each set, and whether it names the stops."""

    root = Path(webserver_package.__file__).parent
    stops = set(POPULATION_APPROVAL_STOPS.values())
    for source in sorted(root.rglob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        parents = {
            child: node
            for node in ast.walk(tree)
            for child in ast.iter_child_nodes(node)
        }
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Set, ast.List, ast.Tuple)):
                continue
            codes = {
                item.value
                for item in node.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            }
            if not codes & _PLAN_REVIEW_CODES:
                continue
            spreads = {
                ast.unparse(item.value)
                for item in node.elts
                if isinstance(item, ast.Starred)
            }
            function, name, parent = "", "", parents.get(node)
            while parent is not None:
                if not name and isinstance(parent, ast.Assign):
                    name = ast.unparse(parent.targets[0])
                if not function and isinstance(
                    parent, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    function = parent.name
                parent = parents.get(parent)
            key = (source.relative_to(root).as_posix(), function or "<module>", name)
            yield key, stops <= codes or _STOPS_SPREAD in spreads


def test_every_set_that_reads_a_plan_review_names_the_stops_or_says_why_not() -> None:
    sets: dict[tuple[str, str, str], bool] = {}
    for key, names_stops in _plan_review_code_sets():
        # Two unnamed sets in one function share a key; both must name them.
        sets[key] = sets.get(key, True) and names_stops

    assert {
        (
            "pi_copilot/workflow.py",
            "build_research_workflow_snapshot",
            "plan_review_codes",
        ),
        ("pi_copilot/projections.py", "project_run_row", "waiting_for_plan_review"),
        ("routes/agent.py", "<module>", "_CANDIDATE_PLAN_WORKFLOW_CODES"),
    } <= {key for key, names_stops in sets.items() if names_stops}
    assert [
        key
        for key, names_stops in sorted(sets.items())
        if not names_stops and key not in _NEED_NOT_NAME_THE_STOPS
    ] == []
    # Each reason still describes a set that exists and omits the stops.
    assert [key for key in _NEED_NOT_NAME_THE_STOPS if sets.get(key) is not False] == []
