"""Host-bound preparation decisions use synthetic, non-patient data only."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from types import SimpleNamespace
import json

import pytest
from pydantic import ValidationError

from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.orchestration.workflow import HumanReviewDecision, HumanReviewRequest
from easyicu.webserver.host_action_contracts import (
    HostActionRequest, HostActionTag, RawPreparationReviewDecision, decision_key, host_action_id,
)
from tests.support.raw_preparation_decision import _host_decision

from easyicu.webserver.raw_preparation_decisions import (
    consume_host_review, record_host_review, verify_host_review,
)




def test_host_review_binds_context_scope_and_cannot_be_replayed(tmp_path, monkeypatch):
    pending, args, binding = _host_decision(tmp_path, monkeypatch)
    checkpoint = record_host_review(**args)
    assert checkpoint.decision.reviewer == "actual-host-operator"
    verify_host_review(checkpoint=checkpoint, **binding)
    for name, replacement in [("project_id", "other"), ("session_id", "other"),
                              ("study_context_id", "other"), ("run_id", "other"),
                              ("scope", {"source": "different"})]:
        with pytest.raises(ValueError, match="host_binding_mismatch"):
            verify_host_review(checkpoint=checkpoint, **{**binding, name: replacement})
    with pytest.raises(ValueError, match="nonce_replayed"):
        record_host_review(**args)
    consume_host_review(checkpoint=checkpoint, **binding)
    with pytest.raises(ValueError, match="nonce_replayed"):
        consume_host_review(checkpoint=checkpoint, **binding)


def test_forged_decision_and_missing_host_action_never_authorize(tmp_path, monkeypatch):
    pending, args, binding = _host_decision(tmp_path, monkeypatch)
    forged = pending.decide(HumanReviewDecision(review_id=pending.request.review_id,
        authority_sha256=pending.request.authority_sha256, decision="approved",
        reviewer="claimed-human", decided_at=datetime.now(timezone.utc).isoformat()))
    with pytest.raises(ValueError, match="host_receipt_invalid"):
        verify_host_review(checkpoint=forged, **binding)
    with pytest.raises(ValueError, match="host_action_unrecorded"):
        record_host_review(**{**args, "job_id": "not-the-recorded-job"})
    with pytest.raises(ValueError, match="host_review_mismatch"):
        record_host_review(**{**args, "run_id": "another-run"})
    with pytest.raises(ValueError, match="host_review_mismatch"):
        record_host_review(**{**args, "scientific_configuration_sha256": "d" * 64})
    with pytest.raises(ValidationError):
        HostActionRequest(session_id="session", project_id="project", action_code="execute_plan",
                          decision=args["host_action"].decision)
    checkpoint = record_host_review(**args)
    # A client rewriting a valid decision and its bare SHA cannot recompute the host HMAC.
    from easyicu import state_paths
    path = state_paths.state_root() / "private-preparation-decisions" / (binding["nonce"] + ".json")
    document = json.loads(path.read_text())
    document["actor"] = "agent"
    document["signature"] = canonical_sha256({k: v for k, v in document.items() if k != "signature"})
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="host_receipt_invalid"):
        verify_host_review(checkpoint=checkpoint, **binding)


def test_concurrent_consumers_cannot_reuse_one_nonce(tmp_path, monkeypatch):
    _, args, binding = _host_decision(tmp_path, monkeypatch)
    checkpoint = record_host_review(**args)
    def consume():
        try:
            consume_host_review(checkpoint=checkpoint, **binding)
            return "consumed"
        except ValueError:
            return "refused"
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sorted(pool.map(lambda _: consume(), range(4))) == ["consumed", "refused", "refused", "refused"]
