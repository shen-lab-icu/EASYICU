from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.orchestration.workflow import HumanReviewRequest
from easyicu.webserver.raw_preparation_projection import pending_preparation_projection


def test_pending_bound_run_is_informational_only(tmp_path):
    scope = {"source": "synthetic-database", "files": ["events.csv"], "purpose": "prepare_only"}
    payload = {"raw_preparation_scope": scope}
    request = HumanReviewRequest.create(kind="scientific_stop", summary="Review events.csv",
                                        payload=payload, authority_sha256=canonical_sha256(payload))
    checkpoint = RawPreparationCheckpoint.pending(plan_sha256="a" * 64, request=request)
    run = {"run_id": "run-1", "project_dir": str(tmp_path)}
    assert pending_preparation_projection(run) is None
    path = tmp_path / "raw_preparation_checkpoint.json"
    path.write_text(checkpoint.model_dump_json())
    projected = pending_preparation_projection(run)
    assert projected["summary"] == "Review events.csv"
    assert projected["scope"] == scope
    assert projected["approvable"] is False
    assert projected["status"] == "awaiting_consumption_closure"
    assert str(tmp_path) not in str(projected)
    # Ambiguous competing checkpoints are never silently selected.
    (tmp_path / "sic_raw_preparation_checkpoint.json").write_text(checkpoint.model_dump_json())
    assert pending_preparation_projection(run) is None
    path.write_text("not a valid checkpoint")
    (tmp_path / "sic_raw_preparation_checkpoint.json").unlink()
    assert pending_preparation_projection(run) is None
