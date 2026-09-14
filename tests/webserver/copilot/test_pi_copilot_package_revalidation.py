"""Prepared-package revalidation at Copilot execution boundaries."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.webserver.copilot import test_pi_copilot_research_workflow as base


def test_pipeline_bridge_rejects_direct_scientific_provider_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export = base._write_pipeline_export(tmp_path / "export")
    monkeypatch.setattr(
        base.research_pipeline_run_preparation,
        "_data_foundation_profile",
        lambda **_kwargs: base._foundation_profile(),
    )

    with pytest.raises(base.agent_pipeline_runs.ResearchPipelineRunError) as exc:
        base.agent_pipeline_runs.make_research_pipeline_run_runner(
            export_path=str(export),
            study_context=base._complete_study(),
            project_root=str(tmp_path / "projects"),
            provider={"provider": "openai", "external": True},
            provider_environment=None,
        )

    assert exc.value.code == "research_pipeline_pi_verified_credentials_required"


def test_pipeline_revalidates_package_before_provider_or_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base._assume_execution_runtime_ready(monkeypatch)
    export = base._write_pipeline_export(tmp_path / "export")
    study = {
        **base._complete_study(),
        "data_source": {"path": str(export), "database": "miiv"},
    }
    monkeypatch.setattr(
        base.research_pipeline_run_preparation,
        "_data_foundation_profile",
        lambda **_kwargs: base._foundation_profile(),
    )
    provider_called = False

    def provider_client(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal provider_called
        provider_called = True
        raise AssertionError("provider must not be reached after package drift")

    monkeypatch.setattr(
        base.provider_adapter,
        "build_research_agent_provider_client",
        provider_client,
    )
    runner = base.agent_pipeline_runs.make_research_pipeline_run_runner(
        export_path=str(export),
        study_context=study,
        project_root=str(tmp_path / "projects"),
        provider={"provider": "openai", "external": True},
        provider_environment=base._PI_PROVIDER_ENVIRONMENT,
        budget_mode="full_reviewed",
    )
    (export / "demographics.parquet").write_bytes(b"changed-after-submit")

    with pytest.raises(base.agent_pipeline_runs.ResearchPipelineRunError) as exc:
        runner(SimpleNamespace(id="job-drift", emit=lambda _event: None))

    assert exc.value.code == "research_pipeline_package_binding_changed"
    assert provider_called is False


def test_plan_approval_revalidates_the_exact_prepared_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base._allow_current_scientific_review(monkeypatch)
    study, package_binding = base._study_with_package_binding(tmp_path / "package")
    request = base.HumanReviewRequest.create(
        kind="scientific_stop",
        summary="Review package-bound plan.",
        authority_sha256="f" * 64,
        payload={"reason": "operator_plan_approval_required"},
    )
    pending = base.HumanReviewPending(
        run_id="run-package-drift",
        thread_id="run-package-drift",
        run_dir=str(tmp_path / "run-package-drift"),
        requests=(request,),
    )
    pipeline_called = False

    class _Pipeline:
        def resume_human_review(self, *_args: Any, **_kwargs: Any) -> Any:
            nonlocal pipeline_called
            pipeline_called = True
            raise AssertionError("drifted package must not reach Pipeline")

    entry = base.agent_pipeline_runs._PendingRun(
        pipeline=_Pipeline(),
        pending=pending,
        wrapper_dir=tmp_path,
        study=study,
        provider={},
        acquisition=SimpleNamespace(),
        created_at=1.0,
        prepared_package_binding=package_binding,
    )
    base._install_pending_review(monkeypatch, entry)
    export = Path(study["data_source"]["path"])
    (export / "demographics.parquet").write_bytes(b"changed-before-approval")

    with pytest.raises(base.agent_pipeline_runs.ResearchPipelineRunError) as exc:
        base.agent_pipeline_runs.resume_research_pipeline(
            run_id=pending.run_id,
            study_context_id=study["id"],
            decision="approved",
            reviewer="server reviewer",
            note="",
            job=SimpleNamespace(emit=lambda _event: None, cancel_requested=False),
            current_study_context=study,
        )

    assert exc.value.code == "research_pipeline_package_binding_changed"
    assert pipeline_called is False
