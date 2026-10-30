"""Host-authenticated sandbox integration using only regenerated synthetic data."""
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.orchestration.workflow import HumanReviewRequest
from easyicu.synthetic_sources import generate_source
from easyicu.webserver.host_action_contracts import (
    HostActionRequest, HostActionTag, RawPreparationReviewDecision, decision_key, host_action_id,
)
from easyicu.webserver.raw_preparation_decisions import RawPreparationHostBinding, record_host_review
from easyicu.webserver.reviewed_snapshot_producer import run_reviewed_snapshot_producer
from tests.support.scoped_synthetic_producer import SyntheticAdapter


@pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("sandbox-exec"), reason="requires macOS sandbox")
def test_host_receipt_consumed_before_real_exporter_runs(tmp_path, monkeypatch):
    from easyicu.webserver import agent_pipeline_runs, jobs
    from easyicu.webserver.pi_copilot import service

    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "host"))
    source = tmp_path / "synthetic"
    manifest = generate_source(source, "sic", stays=10)
    scope = {"adapter_id": "synthetic-test-only", "source": str(source), "database": "sic", "purpose": "synthetic_fixture_only",
             "files": sorted([*manifest["files"], "README.md", "synthetic_source.json"])}
    payload = {"synthetic_preparation_scope": scope}
    request = HumanReviewRequest.create(kind="scientific_stop", summary="Synthetic fixture only",
                                        payload=payload, authority_sha256=canonical_sha256(payload))
    pending = RawPreparationCheckpoint.pending(plan_sha256="c" * 64, request=request)
    choice = RawPreparationReviewDecision(run_id="run", scientific_configuration_sha256="a" * 64,
        checkpoint_sha256=pending.checkpoint_sha256, scope_sha256=canonical_sha256(scope),
        review_authority_sha256=request.authority_sha256, nonce="b" * 64)
    action = HostActionRequest(session_id="session", project_id="project",
                               action_code="review_raw_preparation", decision=choice)
    key = decision_key(choice, review_decision="approved")
    tag = HostActionTag(session_id="session", project_id="project", study_context_id="study",
        action_code=action.action_code, decision_key=key, action_id=host_action_id("session", action.action_code, key))
    monkeypatch.setattr(jobs.MANAGER, "get", lambda job_id: SimpleNamespace(host_action=tag) if job_id == "job" else None)
    monkeypatch.setattr(agent_pipeline_runs, "server_reviewer_identity", lambda: "synthetic-test-reviewer")
    owner = SimpleNamespace(host_action_session=lambda *args, **kwargs: None,
                            host_action_child_job=lambda *args, **kwargs: {"child_job_id": "job"})
    monkeypatch.setattr(service, "get_pi_copilot_service", lambda: owner)
    checkpoint = record_host_review(host_action=action, pending=pending, decision="approved", study_context_id="study",
                                    job_id="job", run_id="run", scientific_configuration_sha256="a" * 64)
    binding = RawPreparationHostBinding(project_id="project", session_id="session", study_context_id="study",
                                        run_id="run", nonce=choice.nonce)
    source_code = Path(__file__).resolve().parents[3] / "src"
    from easyicu.webserver import reviewed_snapshot_producer as producer
    adapter = SyntheticAdapter(source, Path(sys.executable), [Path(sys.prefix), Path(sys.base_prefix), source_code])
    kwargs = dict(checkpoint=checkpoint, binding=binding, scope=scope, host_work_root=tmp_path)
    with pytest.raises(ValueError, match="read_closure_unverified"):
        run_reviewed_snapshot_producer(**kwargs)
    monkeypatch.setitem(producer._CLOSED_ADAPTERS, "synthetic-test-only", adapter)
    with pytest.raises(ValueError, match="host_binding_mismatch"):
        run_reviewed_snapshot_producer(**{**kwargs, "binding": binding.model_copy(update={"run_id": "wrong"})})
    changed_scope = {**scope, "files": [*scope["files"], "extra.parquet"]}
    with pytest.raises(ValueError, match="host_binding_mismatch"):
        run_reviewed_snapshot_producer(**{**kwargs, "scope": changed_scope})
    assert not list(tmp_path.glob("reviewed-producer-*"))
    result, receipt, output = run_reviewed_snapshot_producer(**kwargs)
    assert result.returncode == 0, result.stderr
    assert (output / "_manifest.json").is_file()
    import pandas as pd

    lactate = pd.read_parquet(output / "blood_gas.parquet")
    outcome = pd.read_parquet(output / "sic_case_landmark.parquet")
    assert not lactate.empty and "lact" in lactate.columns
    assert not outcome.empty and "care_unit_to_case_closure_days" in outcome.columns
    assert receipt.public_receipt()["actual_consumption_verified"] is False
    with pytest.raises(ValueError, match="nonce_replayed"):
        run_reviewed_snapshot_producer(**kwargs)
