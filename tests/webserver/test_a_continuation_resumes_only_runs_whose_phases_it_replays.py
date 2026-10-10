"""A Planner continuation resumes only a run whose every phase it performs again.

A continuation replays the source run's checkpoint on a context it rebuilds.
It does not perform again a phase the source run performed before its
Planner -- asking for the study's exposure groupings and staging them is
one -- so the rebuilt context would lack what that phase formed and the
checkpoint, bound to the source's context, could not be resumed.  The
launch owner refuses such a source and names the phases; an automatic seed
then plans anew and says why, while an explicit checkpoint resume reports
the refusal.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from easyicu.research_agent.orchestration.exposure_grouping_phase import (
    EXPOSURE_GROUPINGS_FILENAME,
    EXPOSURE_GROUPINGS_RECORD_SCHEMA,
)
from easyicu.research_agent.planning.exposure_group_compile import (
    compile_exposure_groupings,
)
from easyicu.research_agent.planning.exposure_group_spec import (
    read_stated_exposure_groupings,
)
from easyicu.research_agent.schema import (
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)
from easyicu.webserver import agent_review_recovery as recovery
from easyicu.webserver import research_launch_resume as owner
from easyicu.webserver import research_run_submission as submission
from easyicu.webserver import study_contexts
from tests.research_agent.planning.progressive_planner_fixtures import (
    _context as _planner_context,
)

_REFUSED = "research_pipeline_development_resume_phase_not_replayed"
_REASON = "development_resume_source_phase_not_replayed"


def _lab(name: str, summary: str) -> ConceptDescriptor:
    return ConceptDescriptor(
        name=name,
        role=VariableRole.LAB,
        dtype="float64",
        unit="mg/dL",
        source_concept="glu",
        unit_normalization=f"window_numeric_{summary}",
        analysis_window="icu_admission[0,24]h",
        valid_range=[0, 1000],
    )


def _context() -> ResearchContext:
    base = _planner_context()
    constraints = json.dumps(
        {
            "materialization_window": {
                "role": "outer_observation_window",
                "anchor": "ICU admission",
                "hours": 24.0,
            }
        }
    )
    return base.model_copy(
        update={
            "variables": [
                *base.variables,
                _lab("glu_min", "min"),
                _lab("glu_max", "max"),
            ],
            "data_constraints": constraints,
        }
    )


def _rule(summary: str, op: str, value: float) -> dict[str, Any]:
    return {"summary": summary, "op": op, "value": value, "unit": "mg/dL"}


def _formed_record() -> dict[str, Any]:
    stated = read_stated_exposure_groupings(
        {
            "groupings": [
                {
                    "id": "x1",
                    "concept": "glu",
                    "window": {"start_hours": 0, "end_hours": 24},
                    "scale": "nominal",
                    "groups": [
                        {"id": "g1", "label": "low", "rule": _rule("min", "<", 70)},
                        {"id": "g2", "label": "high", "rule": _rule("max", ">", 180)},
                        {"id": "g3", "label": "in range", "rule": "otherwise"},
                    ],
                    "unmeasured": {"handling": "own_group", "label": "not measured"},
                    "quote": "glucose below 70 or above 180",
                    "source": "question",
                }
            ]
        }
    )
    compiled = compile_exposure_groupings(stated, _context())
    return {
        "schema_version": EXPOSURE_GROUPINGS_RECORD_SCHEMA,
        "exposure_grouping_enabled": True,
        "planner": None,
        "compiled": compiled.record(),
        "compiled_sha256": compiled.sha256(),
    }


_NOT_ASKED = {
    "schema_version": EXPOSURE_GROUPINGS_RECORD_SCHEMA,
    "exposure_grouping_enabled": True,
    "not_asked": "no_value_to_group",
}


@pytest.fixture
def source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A prior planning run with a digest-bound launch scope and a checkpoint."""

    study = {
        "id": "study-phases",
        "question": "Describe glucose on the first ICU day",
        "data_source": {"path": str(tmp_path / "source"), "database": "eicu"},
    }
    wrapper = tmp_path / study["id"] / "run_prior"
    run = wrapper / "pipeline/run_pipeline"
    run.mkdir(parents=True)
    checkpoint = run / "progressive_planner_checkpoint_002.json"
    checkpoint.write_text("fixture: chain validation is the separate existing owner")
    monkeypatch.setattr(
        owner,
        "_development_progressive_resume_binding",
        lambda **kwargs: (checkpoint, "a" * 64),
    )
    seed = recovery.WebReviewRecoverySeed.create(
        wrapper_dir=str(wrapper.resolve()),
        study=study,
        scientific_configuration_sha256=study_contexts.scientific_configuration_sha256(
            study
        ),
        provider_meta={},
        provider_public={},
        credential_source="pi_verified",
        budget_mode="planner_canary",
        prepared_package_binding=None,
        pipeline_config={"bound_plan_revision_contract": None},
        pipeline_config_sha256="c" * 64,
        acquisition_projection={},
        hard_stop_ledger_path="",
        hard_stop_task_id="web-prior",
        hard_stop_declaration_sha256="d" * 64,
        created_at=1.0,
    )
    (wrapper / ".runtime").mkdir()
    (wrapper / ".runtime/web_review_recovery_seed.json").write_text(
        seed.model_dump_json()
    )

    def scope():
        return owner._development_resume_launch_scope(
            project_root=str(tmp_path), study=study, source_job_id="prior"
        )

    return SimpleNamespace(study=study, run=run, scope=scope)


def _write(run: Path, record: Any) -> None:
    raw = record if isinstance(record, str) else json.dumps(record)
    (run / EXPOSURE_GROUPINGS_FILENAME).write_text(raw, encoding="utf-8")


def test_a_run_that_formed_a_grouping_is_no_continuations_source(source) -> None:
    _write(source.run, _formed_record())

    with pytest.raises(owner.ResearchPipelineRunError) as refused:
        source.scope()

    assert refused.value.code == _REFUSED == owner.DEVELOPMENT_RESUME_PHASE_NOT_REPLAYED
    assert refused.value.details == {
        "reason_code": _REASON,
        "phases": ["exposure_grouping"],
    }
    assert "exposure_grouping" in str(refused.value)


@pytest.mark.parametrize("record", [None, _NOT_ASKED], ids=["no_record", "not_asked"])
def test_a_run_that_formed_no_grouping_is_still_resumed(source, record) -> None:
    if record is not None:
        _write(source.run, record)

    assert source.scope().budget_mode == "planner_canary"


@pytest.mark.parametrize("record", ["{", json.dumps({"schema_version": "other"})])
def test_a_grouping_record_that_cannot_be_read_counts_as_formed(source, record) -> None:
    _write(source.run, record)

    with pytest.raises(owner.ResearchPipelineRunError) as refused:
        source.scope()

    assert refused.value.details["phases"] == ["exposure_grouping"]


def test_any_phase_a_continuation_does_not_replay_refuses_its_source(
    source, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rule is the table's, not a special case of the grouping."""

    monkeypatch.setattr(
        owner,
        "_PHASES_A_CONTINUATION_DOES_NOT_REPLAY",
        (("synthetic_phase", lambda run_dir: (run_dir / "synthetic.json").is_file()),),
    )
    _write(source.run, _formed_record())
    assert source.scope().budget_mode == "planner_canary"

    (source.run / "synthetic.json").write_text("{}")
    with pytest.raises(owner.ResearchPipelineRunError) as refused:
        source.scope()
    assert refused.value.details == {
        "reason_code": _REASON,
        "phases": ["synthetic_phase"],
    }


@pytest.fixture
def submit(source, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    study = source.study
    events: list[Any] = []
    audit: list[tuple[str, dict]] = []
    monkeypatch.delenv("EASYICU_DEVELOPMENT_REVIEWED_EXECUTION", raising=False)
    monkeypatch.setattr(submission.context_store, "get_context", lambda _: study)
    monkeypatch.setattr(
        submission.dataio, "describe_export_source", lambda _: {"ok": True}
    )
    monkeypatch.setattr(
        submission.dataio, "prepared_export_manifest_path", lambda _: None
    )
    monkeypatch.setattr(
        submission,
        "build_research_workflow_snapshot",
        lambda **k: SimpleNamespace(planning_prerequisites_missing=[]),
    )
    monkeypatch.setattr(
        submission, "provider_environment_for_agent_run", lambda **k: {}
    )
    monkeypatch.setattr(
        submission.settings_store, "load_settings", lambda: {"ai_enabled": True}
    )
    monkeypatch.setattr(
        submission.capabilities, "validate_compute_target", lambda _: {"ok": True}
    )
    monkeypatch.setattr(
        submission.agent_runs, "resolve_agent_provider_config", lambda **k: {}
    )
    monkeypatch.setattr(
        submission.context_store, "build_agent_context_binding", lambda *a, **k: {}
    )
    monkeypatch.setattr(
        submission,
        "research_pipeline_workspace",
        lambda: SimpleNamespace(project_root=lambda _: tmp_path),
    )
    monkeypatch.setattr(submission, "list_bound_run_history", lambda **k: [])
    monkeypatch.setattr(
        submission, "resumable_planner_checkpoint_job_id", lambda **k: "prior"
    )
    monkeypatch.setattr(
        submission.context_store, "handoff_context", lambda *a, **k: {"revision": 1}
    )
    monkeypatch.setattr(
        submission.capabilities,
        "record_tool_event",
        lambda kind, detail: audit.append((kind, detail)),
    )

    def make_runner(**kwargs):
        events.append(("runner", kwargs))
        return lambda _: {}

    def enqueue(*args):
        events.append("job")
        return SimpleNamespace(id="next-job", kind="agent-run", status="queued")

    monkeypatch.setattr(
        submission.agent_pipeline_runs, "make_research_pipeline_run_runner", make_runner
    )
    monkeypatch.setattr(submission, "_submit_job", enqueue)

    def run(start: str):
        return submission.submit_research_run(
            submission.ResearchRunSubmissionRequest(
                study_context_id=study["id"],
                provider="mock",
                credential_source="pi_verified",
                external_llm_opt_in=True,
                intent="candidate_plan",
                planner_start_mode=start,
            ),
            authorize=lambda: events.append("authorize"),
        )

    return SimpleNamespace(run=run, events=events, audit=audit)


def test_a_retried_plan_whose_seed_formed_a_grouping_plans_anew_and_says_why(
    source, submit
) -> None:
    _write(source.run, _formed_record())

    receipt = submit.run("auto")

    runner = next(event[1] for event in submit.events if isinstance(event, tuple))
    assert "development_resume_source_job_id" not in runner
    assert receipt.resume_source_job_id is None
    declined = {
        "source_job_id": "prior",
        "reason_code": _REASON,
        "phases": ["exposure_grouping"],
    }
    assert receipt.resume_seed_rejected == declined
    [(kind, detail)] = submit.audit
    assert kind == "agent_run_submitted"
    assert detail["development_resume_seed_rejected"] == declined
    assert submit.events[-2:] == ["authorize", "job"]


def test_a_retried_plan_whose_seed_formed_no_grouping_still_resumes(
    source, submit
) -> None:
    _write(source.run, _NOT_ASKED)

    receipt = submit.run("auto")

    runner = next(event[1] for event in submit.events if isinstance(event, tuple))
    assert runner["development_resume_source_job_id"] == "prior"
    assert receipt.resume_source_job_id == "prior"
    assert receipt.resume_seed_rejected is None


def test_an_explicit_checkpoint_resume_past_the_phase_is_refused(
    source, submit
) -> None:
    _write(source.run, _formed_record())

    with pytest.raises(submission.ResearchRunSubmissionError) as refused:
        submit.run("resume_checkpoint")

    assert refused.value.detail["error"] == _REFUSED
    assert refused.value.detail["details"]["phases"] == ["exposure_grouping"]
    assert submit.events == []
    assert submit.audit == []
