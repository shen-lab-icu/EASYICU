from __future__ import annotations

from types import SimpleNamespace

import pytest

from easyicu.research_agent.execution import step_authority_resume


@pytest.mark.parametrize(
    ("execution", "requires_rerun"),
    [
        (
            SimpleNamespace(
                returncode=125,
                timed_out=False,
                outputs_safe_to_collect=True,
                runner_failure_code=None,
            ),
            True,
        ),
        (
            SimpleNamespace(
                returncode=0,
                timed_out=False,
                outputs_safe_to_collect=True,
                runner_failure_code=None,
            ),
            False,
        ),
    ],
)
def test_explicit_resume_reruns_failed_execution_capsule(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    execution,
    requires_rerun: bool,
) -> None:
    selected = SimpleNamespace(capsule=SimpleNamespace(execution=execution))
    monkeypatch.setattr(
        step_authority_resume,
        "load_checkpoint_selected_step_capsule",
        lambda *_args, **_kwargs: selected,
    )
    state = SimpleNamespace(selected_resume_capsule=None)
    record: dict[str, object] = {}
    request = SimpleNamespace(
        run_dir=tmp_path,
        step=SimpleNamespace(step_id="analysis"),
        resume_state={},
        requested_resume_from_step_id="analysis",
        prior_step_record=None,
        prior_attempt_records=(),
        step_attempt_state=state,
        step_record=record,
    )

    step_authority_resume._select_resume_candidate(
        request,
        gate_stamp={"deterministic_gate_fingerprint": "f" * 64},
    )

    assert bool(record.get("explicit_failed_execution_retry")) is requires_rerun
