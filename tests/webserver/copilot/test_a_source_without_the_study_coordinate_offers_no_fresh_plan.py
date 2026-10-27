"""A source without the study's own exposure or outcome offers no fresh plan.

``agent_pipeline_runs`` stops before the model call when no planning-menu
entry resolves to a study coordinate. When the availability owner says the
source cannot provide it, regenerating on the same configuration repeats the
stop, so the workflow offers none -- until the configuration changes or a
build provides it. A menu gap (the source provides it) never holds a plan back.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from easyicu.research_agent import concept_availability
from easyicu.webserver import study_contexts as study_context_owner
from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot
from tests.webserver.copilot.research_workflow_fixtures import complete_study


def _stopped(study, reason_code="study_coordinate_unavailable", digest=None):
    return {
        "run_id": "run-coordinate-absent",
        "run_type": "full",
        "engine": "easyicu.research_agent.pipeline",
        "gate_status": "blocked",
        "gate_reason": "data_foundation_blocked",
        "gate_detail": {"reason_code": reason_code, "missing_concepts": ["synthetic_outcome"]},
        "run_status": "completed",
        "scientific_configuration_sha256": digest
        or study_context_owner.scientific_configuration_sha256(study),
        "artifact_names": ["source_run_manifest.json"],
    }


def _owner_says(monkeypatch, available, reason="no_direct_source_or_dependencies"):
    monkeypatch.setattr(
        concept_availability, "explain_concept_availability",
        lambda **kw: SimpleNamespace(available=available, reason=reason),
    )


def _snapshot(study, run):
    return build_research_workflow_snapshot(
        study=study, active_export_present=True, active_job=None, latest_run=run,
    )


def test_a_source_that_cannot_provide_the_coordinate_offers_no_fresh_plan(monkeypatch) -> None:
    """The HiRID ICU-mortality page offered "regenerate" twice (user 2026-10-27);
    each repeat spent a model call and stopped the same way."""
    _owner_says(monkeypatch, False)
    study = complete_study()
    snapshot = _snapshot(study, _stopped(study))
    by_id = {row.id: row for row in snapshot.stages}
    assert snapshot.next_action_code == "study_coordinate_unavailable"
    assert by_id["plan"].status == "blocked"
    assert all(action.kind not in {"edit_plan", "drop_variable"} for action in snapshot.next_actions)


def test_another_source_or_question_offers_a_fresh_plan_again(monkeypatch) -> None:
    _owner_says(monkeypatch, False)
    study = complete_study()
    snapshot = _snapshot(study, _stopped(study, digest="b" * 64))
    assert snapshot.next_action_code == "failed_pipeline_requires_fresh_plan"


def test_a_build_that_now_provides_the_coordinate_offers_a_fresh_plan(monkeypatch) -> None:
    # Same configuration, but the owner now says the source provides it.
    _owner_says(monkeypatch, True)
    study = complete_study()
    snapshot = _snapshot(study, _stopped(study))
    assert snapshot.next_action_code == "failed_pipeline_requires_fresh_plan"


def test_a_name_the_dictionary_does_not_know_never_holds_the_plan_back(monkeypatch) -> None:
    # A reader defect naming a horizon as the outcome (``icu_mortality_72h``)
    # is unknown on every source; it is EasyICU's gap, not the source's.
    _owner_says(monkeypatch, False, reason="concept_not_found")
    study = complete_study()
    snapshot = _snapshot(study, _stopped(study))
    assert snapshot.next_action_code == "failed_pipeline_requires_fresh_plan"


@pytest.mark.parametrize("available", [True, False])
def test_a_planning_menu_gap_never_holds_the_plan_back(monkeypatch, available) -> None:
    _owner_says(monkeypatch, available)
    study = complete_study()
    snapshot = _snapshot(study, _stopped(study, reason_code="planning_menu_missing_supported_concept"))
    assert snapshot.next_action_code == "failed_pipeline_requires_fresh_plan"
