"""The paused plan's review names each level code of a grouped exposure.

The review a researcher approves shows the groups by the study's own words
(``exposure_group_labels``, read from the run's grouping record by its
owner), says whether the run wrote that record, and shows a record it
cannot read as unreadable rather than as no grouping.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.orchestration.exposure_grouping_phase import (
    EXPOSURE_GROUPINGS_FILENAME,
    EXPOSURE_GROUPINGS_RECORD_SCHEMA,
)
from easyicu.research_agent.orchestration.workflow import (
    HumanReviewPending,
    HumanReviewRequest,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
from easyicu.webserver import agent_pipeline_runs
from tests.webserver.copilot.research_workflow_fixtures import complete_study


def _pending(run_dir: Path) -> HumanReviewPending:
    plan = AnalysisPlan(
        research_question="Is the grouped value associated with death?",
        steps=[
            AnalysisStep(
                step_id="primary",
                intent="Compare the groups",
                method="descriptive",
                inputs=[],
                expected_outputs=["table:groups"],
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
    return HumanReviewPending(
        run_id="run-groupings",
        thread_id="thread-groupings",
        run_dir=str(run_dir),
        requests=(request,),
    )


def _review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    pending = _pending(tmp_path)
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
    return projected


def test_the_review_names_the_levels_its_run_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [{"variable": "glu_group_x1", "status": "labelled"}]
    monkeypatch.setattr(
        agent_pipeline_runs,
        "recorded_exposure_group_labels",
        lambda run_dir: rows if Path(run_dir) == tmp_path else None,
    )
    (tmp_path / EXPOSURE_GROUPINGS_FILENAME).write_text("{}", encoding="utf-8")

    review = _review(tmp_path, monkeypatch)

    assert review["exposure_group_labels"] == rows
    assert review["exposure_group_labels_recorded"] is True


def test_a_run_without_a_record_says_so_and_an_unreadable_one_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    review = _review(tmp_path, monkeypatch)
    assert (
        review["exposure_group_labels"],
        review["exposure_group_labels_recorded"],
    ) == (
        [],
        False,
    )

    (tmp_path / EXPOSURE_GROUPINGS_FILENAME).write_text(
        json.dumps(
            {"schema_version": EXPOSURE_GROUPINGS_RECORD_SCHEMA, "compiled": {}}
        ),
        encoding="utf-8",
    )
    review = _review(tmp_path, monkeypatch)
    assert (
        review["exposure_group_labels"],
        review["exposure_group_labels_recorded"],
    ) == (
        None,
        True,
    )
