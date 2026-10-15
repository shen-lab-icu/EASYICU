from pathlib import Path
import threading
from types import SimpleNamespace
import pytest
from easyicu.research_agent.authority.provider_hard_stop import ProviderHardStopExceeded
from easyicu.research_agent.execution.step_candidate_recovery import StepCandidateRecovery, StepCandidateRecoveryRequest

class _Coder:
    def repair(self, **kwargs):
        raise ProviderHardStopExceeded(code="BATCH_COST_LIMIT", detail="probe")

def test_critic_recovery_preserves_global_hard_stop() -> None:
    findings = []
    request = StepCandidateRecoveryRequest(
        step=SimpleNamespace(step_id="s1"),
        run_dir=Path("/tmp/nonexistent-run"),
        run_id="probe",
        step_current=1,
        total_steps=1,
        requested_resume_from_step_id=None,
        prior_step_record={},
        prior_attempt_records=[],
        provider_budget=SimpleNamespace(),
        step_repair_budget=SimpleNamespace(llm_repair_attempts=0),
        step_record={},
        findings=findings,
        shared_lock=threading.Lock(),
        worker_progress=SimpleNamespace(
            critic_resume_repair_used=False,
            resumed_code_reuse_used=False,
        ),
        quarantine_state=SimpleNamespace(),
        resume_controller=SimpleNamespace(
            prior_negative_critic_report_for_step=lambda step_id: {"status": "negative"},
            prior_code_for_step=lambda step_id: ("print('old')", {"generation_mode": "llm"}),
        ),
        analysis_family="association",
        coder=_Coder(),
        coder_context=SimpleNamespace(),
        coder_authority=SimpleNamespace(),
        step_attempt_state=SimpleNamespace(coordinates=None),
        checkpoint_authority=SimpleNamespace(
            clear_failed_repair_transport=lambda attempt_id: None
        ),
        deterministic_runner_repair_enabled=False,
        emit_progress=lambda *a, **k: None,
        remember_concept_constraints=lambda *a, **k: None,
        consume_llm_repair_budget=lambda *a, **k: True,
        sync_provider_budget=lambda: None,
        authorize_automatic_repair=lambda *a, **k: None,
        record_repair=lambda *a, **k: None,
    )
    
    with pytest.raises(ProviderHardStopExceeded, match="BATCH_COST_LIMIT"):
        StepCandidateRecovery(request).resume_critic_repair_code()
    assert not any(f.validator == "critic_resume_repair" for f in findings)
