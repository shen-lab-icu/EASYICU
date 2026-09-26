"""The Host compiles a reviewed trajectory plan into the study's design.

When the review finds that a plan claiming trajectory classes clusters one
value per ICU stay of coordinates the signed fixed-window owner can model
(``TRAJECTORY_LONGITUDINAL_OWNER_NOT_SEALED``), "apply execution settings"
declares exactly those coordinates as the study's trajectory design.  The
compiler reads the coordinates the review published for the exact plan; it
does not choose them.  Fixtures are generic.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.contracts.trajectory_design import (
    FIXED_WINDOW_TRAJECTORY_DEFAULTS,
)
from easyicu.webserver.pi_copilot.plan_decisions import (
    PlanDecisionError,
    agent_plan_configuration_available,
    compile_agent_plan_configuration,
)
from easyicu.webserver.trajectory_runtime_projection import (
    validate_trajectory_design_declaration,
)

_OWNER = "TRAJECTORY_LONGITUDINAL_OWNER_NOT_SEALED"
_COORDINATES = ["sofa2_resp", "sofa2_cardio", "lact"]


def _study() -> dict:
    return {
        "cohort": {"preset": "all_icu"},
        "confirmations": {"feature_time_window": True},
        "sensitivity_specs": [],
    }


def _facts(coordinates=_COORDINATES, *, executable: bool = True) -> dict:
    return {
        "trajectory_representation": {
            "longitudinal_owner": None,
            "proposed_coordinates": list(coordinates),
            "executable": executable,
        }
    }


def _compile(study=None, codes=(_OWNER,), facts=None):
    return compile_agent_plan_configuration(
        study=study if study is not None else _study(),
        agent_plan={"steps": []},
        runtime_finding_codes=codes,
        patient_cluster_available=False,
        review_facts=_facts() if facts is None else facts,
    )


def test_the_reviewed_coordinates_become_the_study_trajectory_design() -> None:
    compiled = _compile(codes=(_OWNER, "REPEATED_STAY_IDENTITY_UNAVAILABLE"))

    assert compiled.patch["analysis_design"] == {
        "analysis_family": "trajectory_clustering",
        "analysis_unit": "icu_stay",
        "variance_estimator": "model_based",
    }
    design = compiled.patch["trajectory_design"]
    assert design["coordinate_concepts"] == _COORDINATES
    assert {name: design[name] for name in FIXED_WINDOW_TRAJECTORY_DEFAULTS} == dict(
        FIXED_WINDOW_TRAJECTORY_DEFAULTS
    )
    assert compiled.patch["confirmations"]["agent_plan_configuration_compiled"] is True
    # The trajectory owners fit one row per stay: the repeated-stay finding
    # stays a disclosed limitation instead of compiling patient clustering.
    assert compiled.runtime_finding_codes == (_OWNER,)
    assert "cohort" not in compiled.patch
    # The web owner accepts the declaration it will later sign.
    declared = validate_trajectory_design_declaration({**_study(), **compiled.patch})
    assert declared is not None and declared.coordinate_concepts == tuple(_COORDINATES)


def test_availability_follows_the_published_review_facts() -> None:
    def available(facts):
        return agent_plan_configuration_available(
            study=_study(), agent_plan={"steps": []},
            runtime_finding_codes=(_OWNER,), review_facts=facts,
        )

    assert available(_facts())
    assert not available(None)
    assert not available(_facts(executable=False))


@pytest.mark.parametrize(
    ("facts", "code"),
    [
        (None, "agent_plan_trajectory_coordinates_unavailable"),
        ({}, "agent_plan_trajectory_coordinates_unavailable"),
        (_facts(executable=False), "agent_plan_trajectory_coordinates_unavailable"),
        # A tampered or stale fact still has to satisfy the owner's rule.
        (_facts(["sofa_resp", "sofa_cardio"]), "agent_plan_trajectory_coordinates_unavailable"),
        (_facts(["sofa2_resp"]), "agent_plan_trajectory_coordinates_unavailable"),
    ],
)
def test_coordinates_the_review_did_not_publish_are_never_guessed(facts, code) -> None:
    with pytest.raises(PlanDecisionError) as raised:
        compile_agent_plan_configuration(
            study=_study(), agent_plan={"steps": []},
            runtime_finding_codes=(_OWNER,), patient_cluster_available=False,
            review_facts=facts,
        )
    assert raised.value.code == code


def test_a_declared_design_is_not_replaced_by_a_second_one() -> None:
    study = {
        **_study(),
        "analysis_design": {
            "analysis_family": "trajectory_clustering",
            "analysis_unit": "icu_stay",
            "variance_estimator": "model_based",
        },
        "trajectory_design": {"coordinate_concepts": ["sofa2_renal", "sofa2_liver"]},
    }

    with pytest.raises(PlanDecisionError) as raised:
        _compile(study=study)
    assert raised.value.code == "agent_plan_trajectory_design_already_declared"


def test_a_trajectory_design_does_not_absorb_association_coordinates() -> None:
    with pytest.raises(PlanDecisionError) as raised:
        _compile(codes=(_OWNER, "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED"))
    assert raised.value.code == "agent_plan_runtime_finding_unsupported"
    assert raised.value.details == {
        "finding_codes": ["POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED"]
    }


def _changes_required_snapshot(study: dict, facts: dict):
    from easyicu.webserver import study_contexts as study_context_owner
    from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot

    digest = study_context_owner.scientific_configuration_sha256(study)
    review = {
        "status": "changes_required",
        "approval_allowed": False,
        "findings": [
            {
                "code": _OWNER,
                "severity": "blocker",
                "remediation_route": "runtime_capability",
                "message": "The trajectory plan clusters one value per ICU stay.",
                "remediation": "Compile these coordinates.",
            }
        ],
        "facts": {
            **facts,
            "remediation_buckets": {
                "agent_plan_revision": [],
                "runtime_capability": [_OWNER],
                "study_authority_change": [],
                "external_evidence": [],
                "independent_review": [],
            },
        },
    }
    snapshot = build_research_workflow_snapshot(
        study=study,
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_type": "full",
            "run_id": "run-trajectory-candidate",
            "study_id": study["id"],
            "engine": "easyicu.research_agent.pipeline",
            "gate_status": "blocked",
            "gate_reason": "human_plan_review_required",
            "run_status": "human_review_pending",
            "pending_review_reason_codes": ["plan_scientific_changes_required"],
            "scientific_configuration_sha256": digest,
            "artifact_names": [
                "agent_plan.json",
                "scientific_plan_review.json",
                "source_run_manifest.json",
            ],
        },
        plan_review_authority={
            "run_id": "run-trajectory-candidate",
            "resumable_here": True,
            "scientific_configuration_sha256": digest,
            "budget_mode": "planner_canary",
            "research_input_state": "metadata_only",
            "plan_approval_allowed": False,
            "scientific_plan_review": review,
        },
    )
    return snapshot, review


@pytest.mark.parametrize(
    ("facts", "expected"),
    [
        (_facts(), "agent_plan_configuration_required"),
        (_facts(executable=False), "plan_scientific_changes_required"),
        ({}, "plan_scientific_changes_required"),
    ],
    ids=["published", "not_executable", "absent"],
)
def test_the_workflow_offers_the_compile_only_for_published_coordinates(
    facts: dict, expected: str
) -> None:
    from easyicu.webserver.pi_copilot import workflow as workflow_owner
    from tests.webserver.copilot.research_workflow_fixtures import complete_study

    study = complete_study()
    snapshot, review = _changes_required_snapshot(study, facts)
    assert snapshot.next_action_code == "plan_scientific_changes_required"

    enriched = workflow_owner._enrich_plan_review(
        snapshot,
        study=study,
        review={
            "artifact_payloads": {
                "agent_plan.json": {"steps": []},
                "scientific_plan_review.json": review,
            }
        },
    )

    assert enriched.next_action_code == expected


def test_applying_the_settings_declares_the_design_the_review_published(
    tmp_path, monkeypatch
) -> None:
    from types import SimpleNamespace

    from easyicu.webserver import study_contexts
    from easyicu.webserver.pi_copilot import service as service_module
    from easyicu.webserver.pi_copilot.contracts import AuthorityBinding, PiSessionRecord

    monkeypatch.setenv("EASYICU_HOME", str(tmp_path))
    before = {
        "id": "trajectory-study",
        "revision": 1,
        "question": "Which organ-dysfunction trajectory classes emerge early in the ICU stay?",
        "data_source": {"database": "test", "path": "/test/source"},
        **_study(),
    }
    run = {
        "run_id": "trajectory-run",
        "scientific_configuration_sha256": study_contexts.scientific_configuration_sha256(before),
    }
    service = service_module.PiCopilotService.__new__(service_module.PiCopilotService)
    record = PiSessionRecord(
        session_id="trajectory-session",
        project_id="trajectory-project",
        binding=AuthorityBinding(study_context_id=before["id"], study_revision=1, run_id=run["run_id"]),
    )
    current = dict(before)
    monkeypatch.setattr(service, "_scoped_record", lambda *a, **kw: record)
    monkeypatch.setattr(service, "_stale_details", lambda *a: {})
    monkeypatch.setattr(service, "_save_record", lambda *a: None)
    monkeypatch.setattr(
        service,
        "_binding_for_context",
        lambda context, run_id=None: AuthorityBinding(
            study_context_id=context["id"], study_revision=context["revision"], run_id=run_id,
        ),
    )
    monkeypatch.setattr(service_module.study_contexts, "get_context", lambda *a: dict(current))
    monkeypatch.setattr(
        service_module, "list_bound_run_history",
        lambda **kw: [{**run, "project_dir": "/test/trajectory"}],
    )
    monkeypatch.setattr(service_module, "research_pipeline_project_root", lambda *a: "/test")
    monkeypatch.setattr(
        service_module.source_identity_authority,
        "resolve_study_patient_grouping",
        lambda **kw: None,
    )
    monkeypatch.setattr(
        service_module.agent_runs,
        "read_run_review",
        lambda *a: {
            "artifact_payloads": {
                "agent_plan.json": {"steps": []},
                "scientific_plan_review.json": {
                    "facts": {
                        **_facts(),
                        "remediation_buckets": {
                            "runtime_capability": [_OWNER, "REPEATED_STAY_IDENTITY_UNAVAILABLE"]
                        },
                    }
                },
            }
        },
    )
    written = []

    def update(patch, **kw):
        written.append(patch)
        current.update(patch)
        current["revision"] += 1
        return dict(current)

    monkeypatch.setattr(service_module.study_contexts, "upsert_context", update)

    result = service.apply_agent_plan_configuration(
        "trajectory-session", project_id="trajectory-project",
        expected_revision=1, run_id=run["run_id"],
    )

    assert result["next_action"] == "fresh_plan"
    assert result["runtime_finding_codes"] == [_OWNER]
    [patch] = written
    assert patch["trajectory_design"]["coordinate_concepts"] == _COORDINATES
    assert patch["analysis_design"]["analysis_family"] == "trajectory_clustering"
    assert current["question"] == before["question"]
