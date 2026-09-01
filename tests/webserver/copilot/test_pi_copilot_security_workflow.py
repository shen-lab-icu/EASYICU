from __future__ import annotations

from pathlib import Path


WORKFLOW = (
    Path(__file__).parents[1] / ".github" / "workflows" / "pi_workspace_security_ci.yml"
)


def test_pi_security_gate_runs_multiprocess_and_workflow_regressions() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "tests/test_pi_copilot_*.py" in workflow


def test_pi_security_proofs_are_not_skipped_after_scoped_pytest_failure() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    for step_name in (
        "Parse-check Pi sidecar owners",
        "Run JavaScript owner and security contracts",
        "Prove hostile preview origin isolation in Chromium",
    ):
        step = workflow.split(f"- name: {step_name}", maxsplit=1)[1]
        step = step.split("- name:", maxsplit=1)[0]
        assert "if: ${{ !cancelled() }}" in step
