"""A confirmed raw Demo choice follows the 24 h landmark the host adds at launch.

UX leg 3 (2026-11-19): an official-Demo raw plan with a first-day exposure and
in-hospital death always stopped with an internal error. The launch staged the
agent landmark, the runner rechecked the choice on that study, and the receipt,
signed before the landmark, was stale. The choice now follows exactly that
host-derived timing; any other change still makes it stale.
"""

from types import SimpleNamespace

import pytest

from easyicu.webserver import launch_study_updates, study_contexts
from easyicu.webserver.pi_copilot import plan_decisions
from easyicu.webserver.pi_copilot.demo_planning_authority import (
    follow_launch_timing,
    resolve_choice,
)
from easyicu.webserver.pi_copilot.demo_planning_source import (
    resolve_demo_planning_source,
)
from tests.support.official_demo_choice import choice as choice
from tests.support.official_demo_raw import official_raw as official_raw


def _landmark_patch(study):
    return plan_decisions.agent_post_baseline_landmark_patch(study)


def _choose(choice, **extra):
    """The official Demo chosen for a first-day-exposure association study."""
    study = study_contexts.upsert_context({
        "id": choice.study["id"], "analysis_design": {
            "analysis_family": "association_study", "analysis_unit": "icu_stay", "variance_estimator": "model_based",
        }, **extra,
    })
    choice.record.binding = choice.service._binding_for_context(study)
    offer = resolve_demo_planning_source(study)
    choice.service.authorize_data_source(choice.record.session_id, **{
        **choice.body, "expected_revision": study["revision"], "offer_sha256": offer.offer["offer_sha256"],
    })
    study = study_contexts.get_context(study["id"])
    identity = resolve_choice(study)
    assert identity is not None
    return SimpleNamespace(study=study, identity=identity)


@pytest.fixture
def chosen(choice):
    return _choose(choice)


def test_the_launch_landmark_keeps_the_confirmed_choice(chosen):
    patch = _landmark_patch(chosen.study)
    after = {**chosen.study, **patch}
    with pytest.raises(ValueError, match="demo_planning_authority_stale"):
        resolve_choice(after)  # the defect: the receipt predates the landmark
    followed = follow_launch_timing(chosen.study, patch)
    assert followed != chosen.study["demo_planning_authority_ref"]
    assert resolve_choice({**after, "demo_planning_authority_ref": followed}) == chosen.identity
    # A refused launch that relaunches gets the same, single follow.
    assert follow_launch_timing(chosen.study, patch) == followed


@pytest.mark.parametrize("change", [
    "extra_field", "other_spec_id", "other_hours", "other_confirmation", "dropped_row",
    "another_study", "chained_follow", "no_ref",
])
def test_nothing_but_the_hosts_own_landmark_follows(chosen, change):
    study, patch = chosen.study, _landmark_patch(chosen.study)
    row = patch["sensitivity_specs"][-1]
    if change == "extra_field":
        patch = {**patch, "question": "另一个研究问题"}
    elif change == "other_spec_id":
        patch = {**patch, "sensitivity_specs": [*patch["sensitivity_specs"][:-1], {**row, "spec_id": "question_landmark_24h"}]}
    elif change == "other_hours":
        patch = {**patch, "sensitivity_specs": [*patch["sensitivity_specs"][:-1], plan_decisions.landmark_timing_specification(
            outcome="death", landmark_hours=48, spec_id=plan_decisions.AGENT_POST_BASELINE_LANDMARK_SPEC_ID)]}
    elif change == "other_confirmation":
        patch = {**patch, "confirmations": {**patch["confirmations"], "plan_timing_time_varying": True}}
    elif change == "dropped_row":
        patch = {**patch, "sensitivity_specs": []}
    elif change == "another_study":
        study = {**study, "id": "different-study"}
    elif change == "chained_follow":
        followed = follow_launch_timing(study, patch)
        study = {**study, **patch, "demo_planning_authority_ref": followed, "sensitivity_specs": []}
        patch = _landmark_patch(study)
    else:
        study = {**study, "demo_planning_authority_ref": ""}
    with pytest.raises(ValueError, match="demo_planning_authority_stale"):
        follow_launch_timing(study, patch)


def test_a_study_that_already_has_a_timing_row_does_not_follow(choice):
    """Carbon b31107 minor 2: a valid receipt on a study with a timing row still refuses."""
    row = plan_decisions.landmark_timing_specification(
        outcome="death", landmark_hours=plan_decisions.QUESTION_LANDMARK_HOURS, spec_id="question_landmark_24h",
    )
    chosen = _choose(choice, sensitivity_specs=[row])
    assert resolve_choice(chosen.study) == chosen.identity  # the receipt itself is valid
    with pytest.raises(ValueError, match="demo_planning_authority_stale"):
        follow_launch_timing(chosen.study, _landmark_patch(chosen.study))


def test_no_client_path_writes_the_followed_choice(chosen):
    """Only host code passes the authority flag; a request body cannot carry it."""
    from fastapi import HTTPException

    from easyicu.webserver.routes.study_contexts import post_study_context

    followed = follow_launch_timing(chosen.study, _landmark_patch(chosen.study))
    for body in (
        {"id": chosen.study["id"], "demo_planning_authority_ref": followed},
        {"id": chosen.study["id"], "demo_planning_authority_ref": followed, "_server_demo_planning_authority_write": True},
    ):
        with pytest.raises(HTTPException):
            post_study_context({**body, "expected_revision": chosen.study["revision"]})
        with pytest.raises(study_contexts.StudyContextError):
            study_contexts.validate_context_update(body, current_context=chosen.study)
    assert study_contexts.get_context(chosen.study["id"])["demo_planning_authority_ref"] == (
        chosen.study["demo_planning_authority_ref"]
    )


def test_only_the_hosts_launch_landmark_calls_the_follow():
    import ast
    from pathlib import Path

    import easyicu

    users = set()
    for source in Path(easyicu.__file__).parent.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        scopes = {id(child): node.name for node in ast.walk(tree)
                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for child in ast.walk(node)}
        for node in ast.walk(tree):
            named = (
                (isinstance(node, ast.Name) and node.id == "follow_launch_timing")
                or (isinstance(node, ast.Attribute) and node.attr == "follow_launch_timing")
                or (isinstance(node, ast.alias) and "follow_launch_timing" in (node.name, node.asname))
            )
            if named:
                users.add((source.name, scopes.get(id(node), "<module>")))
    assert users == {("plan_decisions.py", "record_post_baseline_landmark")}


def test_a_damaged_receipt_at_launch_is_the_typed_stale_refusal(monkeypatch, tmp_path):
    """Carbon b31107 minor 1: a receipt that cannot be read refuses launch, never crashes it."""
    from easyicu.webserver import research_pipeline_run_preparation as preparation
    from easyicu.webserver.pi_copilot import demo_planning_authority
    from tests.support.first_day_association_launch import (
        _study,
        prepare_first_day_launch,
    )

    prepare, _coordinates = prepare_first_day_launch(monkeypatch, tmp_path)
    study = {**_study(), "revision": 1, "demo_planning_authority_ref": "a" * 64,
             "data_source": {"database": "eicu", "path": str(tmp_path / "synthetic-source")}}

    def unreadable(_ref):
        raise FileNotFoundError("receipt")

    monkeypatch.setattr(demo_planning_authority, "_signed_body", unreadable)
    with pytest.raises(preparation.ResearchPipelineRunError) as raised:
        prepare(study)
    assert raised.value.code == "research_pipeline_demo_planning_authority_stale"


def test_a_staged_launch_hands_the_runner_and_the_store_a_choice_that_resolves(chosen, monkeypatch):
    from easyicu.research_agent.research_context import builder

    study = chosen.study
    monkeypatch.setattr(builder, "build_research_context", lambda **kwargs: None)
    monkeypatch.setattr(plan_decisions, "post_baseline_landmark_configuration",
                        lambda *, study, context: plan_decisions.agent_post_baseline_landmark_patch(study))

    def submit(request):
        timed = plan_decisions.record_post_baseline_landmark(
            study=study, materialization_study=study, primary_exposure="lactate_max_day1",
            target="death", endpoint=object(), database=study["data_source"]["database"],
        )
        # What the runner rechecks before metadata-only planning.
        assert resolve_choice(timed) == chosen.identity
        launch_study_updates.handoff_launch_context(
            study["id"], active_job_id="job-landmark", expected_revision=study["revision"],
        )
        return timed

    timed = launch_study_updates.stage_launch_updates(submit)(SimpleNamespace(study_context_id=study["id"]))
    stored = study_contexts.get_context(study["id"])
    assert stored["demo_planning_authority_ref"] == timed["demo_planning_authority_ref"]
    assert resolve_choice(stored) == chosen.identity
    assert [row["spec_id"] for row in stored["sensitivity_specs"]] == [
        plan_decisions.AGENT_POST_BASELINE_LANDMARK_SPEC_ID,
    ]


def test_a_later_researcher_change_still_clears_the_followed_choice(chosen):
    followed = follow_launch_timing(chosen.study, _landmark_patch(chosen.study))
    study_contexts.upsert_context({
        "id": chosen.study["id"], **_landmark_patch(chosen.study), "demo_planning_authority_ref": followed,
    }, _server_demo_planning_authority_write=True)
    study_contexts.upsert_context({"id": chosen.study["id"], "question": "换一个问题"})
    assert resolve_choice(study_contexts.get_context(chosen.study["id"])) is None
