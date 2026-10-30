from types import SimpleNamespace

import pytest


def test_desktop_plan_refuses_unowned_step_before_review():
    from easyicu.research_agent.execution.owner_declaration import (
        require_host_rendered_plan,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    plan = AnalysisPlan(
        research_question="Describe synthetic observations",
        steps=[
            AnalysisStep(
                step_id="custom",
                intent="Synthetic calculation",
                method="custom algorithm",
                expected_outputs=["table:custom"],
            )
        ],
    )
    with pytest.raises(ValueError, match="research_pipeline_plan_step_needs_sandbox"):
        require_host_rendered_plan(plan=plan)


@pytest.mark.parametrize("docker_ready", [True, False])
def test_desktop_profile_selected_only_without_docker(monkeypatch, docker_ready):
    from easyicu.research_agent.execution import runner
    from easyicu.research_agent.orchestration.profiles import get_submission_profile
    from easyicu.webserver.research_launch_runtime import (
        _submission_profile_ref,
        _require_execution_runtime,
    )

    monkeypatch.setenv("EASYICU_DEPLOYMENT", "desktop")
    monkeypatch.delenv("EASYICU_HOSTED", raising=False)
    monkeypatch.setattr(
        runner,
        "probe_runner_availability",
        lambda kind, **kw: runner.RunnerAvailability(
            kind=kind,
            available=docker_ready if kind == "docker" else True,
            image="synthetic:image" if kind == "docker" else "",
            reason_code="docker_executable_missing",
        ),
    )
    kind = _require_execution_runtime(
        budget_mode="full_reviewed", runner_image="synthetic:image"
    )
    profile = get_submission_profile(
        _submission_profile_ref(
            budget_mode="full_reviewed", live_pubmed=False, runner_kind=kind
        )
    )
    assert profile.requires_runner == ("docker" if docker_ready else "host")
    if not docker_ready:
        assert profile.execution_tier == "first_result"
        assert profile.as_pipeline_options()["enable_latex"] is False
        assert profile.as_pipeline_options()["generate_manuscript"] is False


@pytest.mark.parametrize(
    "entrypoint",
    [
        "coder",
        "resume_critic_repair_code",
        "resume_deterministic_repair_code",
        "preflight_resumed_code",
        "quarantined_resume_draft",
        "capsule",
        "typed_preflight_no_code",
    ],
)
@pytest.mark.parametrize("runner_kind", ["host", "docker"])
@pytest.mark.parametrize("tier_source", ["configuration", "approved_plan"])
def test_initial_model_entrypoints_stop_before_coder_or_capsule(entrypoint, runner_kind, tier_source):
    import inspect
    from easyicu.research_agent.execution.phase_support import (
        _step_resolve_initial_code,
    )
    from contextlib import nullcontext

    args = dict.fromkeys(inspect.signature(_step_resolve_initial_code).parameters)
    flushed = []
    args.update(
        pipeline=SimpleNamespace(_runner_kind=runner_kind, _config=SimpleNamespace(
            execution_tier="first_result" if tier_source == "configuration" else "article")),
        plan_result=SimpleNamespace(plan=SimpleNamespace(design_selection=SimpleNamespace(
            execution_tier="first_result" if tier_source == "approved_plan" else "article"))),
        step=SimpleNamespace(step_id="synthetic"),
        step_record={"step_id": "synthetic"},
        shared_lock=nullcontext(), findings=[], per_step_records=[],
        _append_terminal_step_record=lambda records, record: records.append(record),
        _flush_partial_manifest=lambda: flushed.append(True),
        emit_progress=lambda *a, **kw: None,
    )
    if entrypoint in args:
        args[entrypoint] = object()
    if entrypoint == "capsule":
        args["step_attempt_state"] = SimpleNamespace(selected_resume_capsule=object())
    if entrypoint == "typed_preflight_no_code":
        args["standard_executor"] = SimpleNamespace(analysis_kind="typed_summary")
    code, terminal = _step_resolve_initial_code(**args)
    assert code is None
    assert terminal is args["step_record"]
    assert terminal["status"] == "blocked_host_code_policy"
    assert terminal["runner_failure_code"] == "host_runner_refuses_model_code"
    assert args["per_step_records"] == [terminal]
    assert flushed == [True]
    assert args["findings"][0].detail["reason"] == "host_runner_refuses_model_code"


def test_fresh_typed_code_preserves_provenance_through_candidate_construction():
    from contextlib import nullcontext
    import inspect
    from easyicu.research_agent.execution.phase_support import (
        _step_resolve_initial_code,
    )
    from easyicu.research_agent.execution.candidate_loop import _CandidateLoopState
    from easyicu.research_agent.execution.step_execution import CodeOrigin

    args = dict.fromkeys(inspect.signature(_step_resolve_initial_code).parameters)
    record = {}
    args.update(
        pipeline=SimpleNamespace(_runner_kind="host"),
        step=SimpleNamespace(step_id="synthetic"),
        step_record=record,
        preflight_standard_code="value = 1",
        standard_executor=SimpleNamespace(analysis_kind="synthetic"),
        shared_lock=nullcontext(),
        findings=[],
        step_attempt_state=SimpleNamespace(
            selected_resume_capsule=SimpleNamespace(candidate_code="model code")
        ),
    )
    code, terminal = _step_resolve_initial_code(**args)
    assert code == "value = 1" and terminal is None
    state = _CandidateLoopState.from_initial_code(
        code,
        record,
        concept_approved_code_digest=None,
        deterministic_contract_approved_code_digest=None,
        final_concept_gate_approved_code_digest=None,
    )
    assert state.execution_code_origin() == CodeOrigin.HOST_RENDERED_EXECUTOR
    state.code = "value = 2"
    assert state.execution_code_origin() == CodeOrigin.REPAIR


@pytest.mark.parametrize(
    "origin,mutated",
    [
        ("model_generated", False),
        ("repair", False),
        ("recovered_capsule", False),
        ("host_rendered_executor", True),
    ],
)
@pytest.mark.parametrize("runner_kind", ["host", "docker"])
@pytest.mark.parametrize("tier_source", ["configuration", "approved_plan"])
def test_candidate_refusal_precedes_execution_and_capsule_replay(origin, mutated, runner_kind, tier_source):
    from contextlib import nullcontext
    from easyicu.research_agent.execution.candidate_loop import (
        _CandidateLoopState,
        _candidate_execute_transition,
        _CandidateLoopAction,
    )
    from easyicu.research_agent.execution.step_execution import CodeOrigin

    host = SimpleNamespace(
        pipeline=SimpleNamespace(_runner_kind=runner_kind, _config=SimpleNamespace(
            execution_tier="first_result" if tier_source == "configuration" else "article")),
        plan=SimpleNamespace(design_selection=SimpleNamespace(execution_tier="first_result" if tier_source == "approved_plan" else "article")),
        shared_lock=nullcontext(),
        findings=[],
        per_step_records=[],
        _flush_partial_manifest=lambda: None,
        _append_terminal_step_record=lambda records, record: records.append(record),
    )
    attempt = SimpleNamespace(step=SimpleNamespace(step_id="synthetic"), step_record={})
    state = _CandidateLoopState(
        code="print('rewritten')" if mutated else "print('original')",
        host_rendered_code="print('original')",
        code_origin=CodeOrigin(origin),
        concept_approved_code_digest=None,
        deterministic_contract_approved_code_digest=None,
        final_concept_gate_approved_code_digest=None,
    )
    assert (
        _candidate_execute_transition(host, attempt, state)
        == _CandidateLoopAction.RETURN
    )
    assert host.findings[0].detail["reason"] == "host_runner_refuses_model_code"
    assert host.per_step_records[0]["status"] == "blocked_host_code_policy"


@pytest.mark.parametrize(
    "origin", [None, "llm", "model_generated", "repair", "recovered_capsule"]
)
def test_robustness_cannot_replay_non_host_primary(monkeypatch, tmp_path, origin):
    from easyicu.research_agent.execution.runners.deterministic_robustness import (
        _replay_primary_model_on_frame,
    )

    monkeypatch.setenv("EASYICU_HOST_RENDERED_ONLY", "1")
    result = _replay_primary_model_on_frame(
        spec=SimpleNamespace(spec_id="synthetic", axis="missingness"),
        source={"record": {"code_origin": origin}},
        variant_cohort=None,
        out_dir=tmp_path,
        replay_root=tmp_path / "replay",
        note_prefix="synthetic",
    )
    assert result["error"] == "host_runner_refuses_model_code"
    assert not (tmp_path / "replay").exists()


@pytest.mark.parametrize("desktop,frozen", [(True, True), (True, False), (False, False)])
def test_robustness_replay_keeps_desktop_cache_and_docker_policy(
    monkeypatch, tmp_path, desktop, frozen
):
    import hashlib
    import sys

    import pandas as pd
    from easyicu.research_agent.execution.runners import (
        deterministic_robustness as owner,
    )

    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "run-font-cache"))
    monkeypatch.setattr(sys, "frozen", frozen, raising=False)
    script = tmp_path / "primary.py"
    script.write_text("print('typed primary')\n")
    if desktop:
        monkeypatch.setenv("EASYICU_HOST_RENDERED_ONLY", "1")
    else:
        monkeypatch.delenv("EASYICU_HOST_RENDERED_ONLY", raising=False)
    monkeypatch.setattr(owner, "variant_typed_manifest_path", lambda **kw: None)
    captured = []

    def run(command, **kwargs):
        captured.append((command, kwargs["env"]))
        return SimpleNamespace(returncode=1, stdout="", stderr="synthetic failure")

    monkeypatch.setattr(owner.subprocess, "run", run)
    result = owner._replay_primary_model_on_frame(
        spec=SimpleNamespace(spec_id="synthetic", axis="missingness"),
        source={
            "record": {
                "code_origin": "host_rendered_executor"
                if desktop
                else "model_generated"
            },
            "script_path": script,
            "source_root": tmp_path,
            "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        },
        variant_cohort=pd.DataFrame({"measurement": [1.0, 2.0]}),
        out_dir=tmp_path,
        replay_root=tmp_path / "replay",
        note_prefix="synthetic",
    )
    assert result["error"] == "registered primary-model replay returned exit code 1"
    assert len(captured) == 1
    assert captured[0][1]["MPLCONFIGDIR"] == str(
        tmp_path / ("run-font-cache" if desktop else "replay/matplotlib")
    )
    if frozen:
        assert captured[0][0] == [
            sys.executable, "--easyicu-host-worker", "replay", "--script", str(tmp_path / "replay" / "primary_analysis.py"),
            "--sha256", hashlib.sha256(script.read_bytes()).hexdigest(),
        ]
    else:
        assert captured[0][0] == [sys.executable, str(tmp_path / "replay" / "primary_analysis.py")]


def test_desktop_write_skips_every_manuscript_service(tmp_path):
    from easyicu.research_agent.reporting.write_phase import run_write_phase

    result = run_write_phase(
        SimpleNamespace(_config=SimpleNamespace(generate_manuscript=False)),
        plan_result=None,
        execute_result=None,
        run_dir=tmp_path,
        run_id="synthetic",
        stop_after_analysis=False,
        manuscript_title=None,
        manuscript_authors=None,
        run_language="en",
        emit_progress=lambda *a, **kw: pytest.fail("writer work"),
        force_writer_probe=True,
    )
    assert "results_report.md" in result.bound_path.read_text()
    assert not list(tmp_path.glob("*.tex"))
    assert not list(tmp_path.glob("*.bib"))


def test_desktop_runtime_revision_keeps_approved_config(tmp_path):
    from easyicu.research_agent.orchestration.config import PipelineConfig
    from easyicu.research_agent.orchestration.runtime_revision import (
        prepare_execution_runtime_revision,
    )

    config = PipelineConfig(workdir=tmp_path, runner_kind="host")
    actual, revision = prepare_execution_runtime_revision(
        approved_config=config,
        runner_image="different:docker",
        run_dir=tmp_path,
    )
    assert actual is config and revision is None


def test_host_plan_allows_a_typed_owner_and_refuses_bare_ordinal():
    from easyicu.research_agent.execution.owner_declaration import (
        require_host_rendered_plan,
    )
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    typed = AnalysisStep(
        step_id="summary",
        intent="Summarize synthetic measurements",
        method="descriptive_cohort_summary",
        inputs=["artifact:analysis_cohort", "measurement"],
        expected_outputs=["table:cohort_summary"],
    )
    require_host_rendered_plan(
        plan=AnalysisPlan(research_question="Describe measurements", steps=[typed])
    )
    bare = AnalysisStep(
        step_id="ordinal",
        intent="Test ordered trend",
        method="prespecified_ordinal_trend",
        inputs=["artifact:analysis_cohort", "measurement", "response"],
        expected_outputs=["test:ordinal_trend"],
    )
    with pytest.raises(ValueError, match="research_pipeline_plan_step_needs_sandbox"):
        require_host_rendered_plan(
            plan=AnalysisPlan(
                research_question="Describe measurements", steps=[typed, bare]
            )
        )
    from tests.support.host_runner_cases import (
        ordered_stratified_step,
        ordered_stratified_plan,
    )

    require_host_rendered_plan(plan=ordered_stratified_plan(ordered_stratified_step()))


def test_host_refusal_precedes_human_approval_and_survives_web_projection():
    from easyicu.research_agent.pipeline import _pipeline_run___human_review_invoker
    from easyicu.research_agent.execution.owner_declaration import HostPlanStepNeedsSandbox
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
    from easyicu.webserver.agent_pipeline_runs import _safe_pipeline_typed_failure

    pipeline = SimpleNamespace(
        _runner_kind="host",
        _scientific_runtime_projection_sha256=None,
        _scientific_runtime_authorities=SimpleNamespace(
            trajectory=None, current_case=None
        ),
    )
    plan_result = SimpleNamespace(
        plan=AnalysisPlan(
            research_question="Synthetic calculation",
            steps=[
                AnalysisStep(
                    step_id="custom", intent="Compute a custom result", method="custom"
                )
            ],
        )
    )
    reviewed = []
    with pytest.raises(HostPlanStepNeedsSandbox) as error:
        _pipeline_run___human_review_invoker(
            plan_result, reviewed_plan=reviewed, self=pipeline
        )
    assert reviewed == []
    assert (
        _safe_pipeline_typed_failure(error.value)["reason_code"]
        == "research_pipeline_plan_step_needs_sandbox"
    )


def test_host_config_is_additive_and_cannot_mix_profiles(tmp_path):
    from easyicu.research_agent.orchestration.config import PipelineConfig
    from easyicu.research_agent.orchestration.profiles import DESKTOP_FIRST_RESULT

    options = DESKTOP_FIRST_RESULT.pipeline_options()
    good = PipelineConfig(workdir=tmp_path, runner_kind="host", **options)
    assert good.generate_manuscript is False
    restored = PipelineConfig.from_recovery_payload(
        good.recovery_payload(), expected_digest=good.canonical_digest()
    )
    assert restored.canonical_payload() == good.canonical_payload()
    assert (
        "generate_manuscript"
        not in PipelineConfig(workdir=tmp_path).canonical_payload()
    )
    assert "execution_tier" not in PipelineConfig(workdir=tmp_path).canonical_payload()
    with pytest.raises(ValueError, match="desktop profile requires"):
        PipelineConfig(workdir=tmp_path, runner_kind="docker", **options)
    with pytest.raises(ValueError, match="generate_manuscript"):
        PipelineConfig(
            workdir=tmp_path,
            runner_kind="host",
            **{**options, "generate_manuscript": True},
        )
    with pytest.raises(ValueError, match="paper-facing"):
        PipelineConfig(
            workdir=tmp_path, runner_kind="host", submission_profile_name="npj_dm"
        )
    with pytest.raises(ValueError, match="network"):
        PipelineConfig(workdir=tmp_path, runner_kind="host", runner_network="bridge")


def test_host_pipeline_builds_host_runner(tmp_path):
    import pandas as pd
    from easyicu.research_agent.pipeline import ResearchAgentPipeline
    from easyicu.research_agent.execution.host_runner import HostRunner
    from easyicu.research_agent.orchestration.config import PipelineConfig

    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame({"measurement": [1, 2]}).to_parquet(cohort)
    pipeline = ResearchAgentPipeline(
        config=PipelineConfig(workdir=tmp_path, runner_kind="host", enable_memory=False)
    )
    runner = pipeline._build_runner(
        run_dir=tmp_path, cohort_path=cohort, target_outcome=None
    )
    assert isinstance(runner, HostRunner)


def test_host_retry_uses_kernel_identity_without_inspecting_docker(monkeypatch):
    from easyicu.research_agent.execution.budget_epoch import runtime_attempt_identity
    from easyicu.research_agent.execution.retry_basis import FailedStepRetryBasis
    from easyicu.webserver import execution_retry

    receipt = {
        "schema": "easyicu.host_runtime_preflight/1",
        "provenance": {
            "runtime": "host",
            "execution_kernel_identity_sha256": "a" * 64,
        },
    }
    identity = runtime_attempt_identity(receipt)
    monkeypatch.setattr(
        execution_retry, "current_attempt_identity", lambda **kw: identity
    )
    basis = FailedStepRetryBasis(
        step_id="synthetic",
        status="execution_failed",
        failure_class="code",
        repair_attempts=2,
        repair_limit=2,
        provider_calls=2,
        provider_limit=2,
        repair_budget_exhausted=True,
        identity=identity,
    )
    result = execution_retry.assess_failed_step(
        basis, read_image_id=lambda: pytest.fail("host retry inspected Docker")
    )
    assert result.state == "futile"
    assert result.image_checked is False


def test_concept_repair_revokes_host_provenance_before_candidate_construction(
    monkeypatch,
):
    from contextlib import nullcontext
    from easyicu.research_agent.execution import concept_repair
    from easyicu.research_agent.schema import ValidationFinding

    original, repaired = "value = 1", "value = 2"
    record = {"code_origin": "host_rendered_executor"}
    quarantine = SimpleNamespace(draft_active=False, resumed_draft_used=False)
    worker = SimpleNamespace(
        critic_resume_repair_used=False, deterministic_standard_executor_used=False
    )
    runtime = SimpleNamespace(
        worker_progress=worker,
        quarantine_state=quarantine,
        step_record=record,
        provider_budget=None,
        usage_auditor=SimpleNamespace(name="concept"),
        emit_progress=lambda *a, **kw: None,
        run_id="synthetic",
        step_current=1,
        total_steps=1,
    )
    audit = SimpleNamespace(
        runtime=runtime,
        authority=SimpleNamespace(
            step=SimpleNamespace(step_id="synthetic"), plausibility_scope=None
        ),
        findings_for_code=lambda code, **kw: (
            [
                ValidationFinding(
                    validator="concept",
                    severity="error",
                    message="Synthetic repair required",
                )
            ]
            if code == original
            else []
        ),
    )
    monkeypatch.setattr(
        concept_repair, "host_plausibility_receipt_injected", lambda code, **kw: code
    )
    monkeypatch.setattr(
        concept_repair,
        "flag_only_plausibility_obligation_findings",
        lambda *a, **kw: [],
    )
    request = SimpleNamespace(
        initial_code=original,
        concept_audit=audit,
        standard_executor=None,
        sealed_renderer_state=SimpleNamespace(repair_id=None),
        shared_lock=nullcontext(),
        findings=[],
        services=SimpleNamespace(
            authorized_deterministic_repair=lambda **kw: (
                repaired,
                ["synthetic_repair"],
            ),
            record_repair=lambda **kw: None,
        ),
    )
    result = concept_repair.run_concept_repair_loop(request)
    assert result.code == repaired
    assert record["code_origin"] == "repair"


def test_missing_both_runtimes_refuses_launch(monkeypatch):
    from easyicu.research_agent.execution import runner
    from easyicu.webserver.research_launch_runtime import (
        _require_execution_runtime,
        ResearchPipelineRunError,
    )

    monkeypatch.setenv("EASYICU_DEPLOYMENT", "desktop")
    monkeypatch.delenv("EASYICU_HOSTED", raising=False)
    monkeypatch.setattr(
        runner,
        "probe_runner_availability",
        lambda kind, **kw: runner.RunnerAvailability(
            kind=kind,
            available=False,
            image="",
            reason_code="docker_executable_missing"
            if kind == "docker"
            else "host_runtime_unavailable",
        ),
    )
    with pytest.raises(ResearchPipelineRunError) as error:
        _require_execution_runtime(
            budget_mode="full_reviewed", runner_image="synthetic:image"
        )
    assert error.value.details["reason_code"] == "host_runtime_unavailable"
