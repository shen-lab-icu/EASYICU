from types import SimpleNamespace
import pytest
from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopLedger
from easyicu.research_agent.authority.filesystem import AuthorityFilesystemError
from easyicu.webserver import (
    metadata_provider_client,
    provider_adapter,
    research_job_recovery,
)
from easyicu.webserver.jobs import Job


@pytest.mark.parametrize(
    "mode", ["same", "different", "tampered", "symlink", "directory"]
)
def test_controlled_candidate_rebinds_only_identical_safe_lineage(
    tmp_path, monkeypatch, mode
):
    ledger = ProviderHardStopLedger(
        path=(tmp_path / "study" / "run_old" / "budget.json").resolve(),
        task_ids=("task",),
        limits=provider_adapter.web_research_agent_hard_stop_limits("planner_canary"),
        batch_id="task",
        declaration_sha256="a" * 64,
    )
    task = ledger.start_task("task")
    task.pause()
    wrapper = tmp_path / "study" / "run_candidate"
    wrapper.mkdir(parents=True)
    monkeypatch.setattr(
        research_job_recovery, "_historical_worker_identity", lambda _: "b" * 64
    )
    monkeypatch.setattr(
        research_job_recovery.study_contexts,
        "scientific_configuration_sha256",
        lambda _: "a" * 64,
    )
    env = {
        "OPENAI_API_KEY": "synthetic-secret",
        "OPENAI_BASE_URL": "http://127.0.0.1:1/v1",
        "OPENAI_MODEL": "synthetic",
        "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
        "EASYICU_PROVIDER_DATA_HANDLING": "user_accepted_external",
    }
    provider = {"provider": "openai", "external": True}
    raw, public = provider_adapter.build_research_agent_provider_client(
        provider, environ=env, request_timeout=30
    )
    guarded = metadata_provider_client.MetadataProviderClient(
        inner=raw,
        task=task,
        provider=provider,
        environment=env,
        public=public,
        timeout=30,
    )

    def runner(job):
        recovery = research_job_recovery.recovery_for_job(job)
        research_job_recovery.bind_provider_recovery(
            job, wrapper_dir=wrapper, task=task, client=raw
        )
        file = wrapper / ".runtime" / "controlled_candidate_lineage.json"
        original = file.read_bytes()
        inode = file.stat().st_ino
        ledger_before = ledger.snapshot()
        if mode == "different":
            recovery.controlled_candidate_lineage["original_job_id"] = "different"
        if mode == "tampered":
            file.write_bytes(original[:-2] + b"X" + original[-1:])
        if mode == "symlink":
            elsewhere = tmp_path / "other.json"
            elsewhere.write_bytes(original)
            file.unlink()
            file.symlink_to(elsewhere)
        if mode == "directory":
            file.unlink()
            file.mkdir()
        if mode == "same":
            research_job_recovery.bind_provider_recovery(
                job, wrapper_dir=wrapper, task=task, client=guarded
            )
            assert recovery.client is guarded and recovery.task is task
            assert file.read_bytes() == original and file.stat().st_ino == inode
        else:
            with pytest.raises((ValueError, AuthorityFilesystemError, OSError)):
                research_job_recovery.bind_provider_recovery(
                    job, wrapper_dir=wrapper, task=task, client=guarded
                )
            assert recovery.client is raw
            if mode == "different":
                assert file.read_bytes() == original and file.stat().st_ino == inode
            if mode == "tampered":
                assert file.read_bytes() == original[:-2] + b"X" + original[-1:]
            if mode == "symlink":
                assert file.is_symlink() and elsewhere.read_bytes() == original
            if mode == "directory":
                assert file.is_dir()
        assert ledger.snapshot() == ledger_before
        assert ledger_before["tasks"][0]["calls"] == []
        task.pause()
        return {"gate": {"status": "pending"}}

    request = SimpleNamespace(
        no_automatic_retries=True,
        allow_environment_resume=False,
        study_context={"id": "s"},
    )
    result = research_job_recovery.run_controlled_candidate(
        runner=runner,
        request=request,
        job=Job("candidate", "agent-run"),
        task=task,
        original_job_id="old",
        historical_worker_tree="c" * 40,
    )
    assert result == {"gate": {"status": "pending"}}
