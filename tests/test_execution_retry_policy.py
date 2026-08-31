from __future__ import annotations

import pytest

from easyicu.webserver.execution_retry import (
    preserves_approved_execution_checkpoint,
)


@pytest.mark.parametrize(
    "gate_reason",
    [
        "research_agent_pipeline_failed_closed",
        "research_pipeline_execution_failed",
    ],
)
def test_execution_failures_preserve_the_approved_checkpoint(gate_reason: str) -> None:
    assert preserves_approved_execution_checkpoint(gate_reason) is True


@pytest.mark.parametrize(
    "gate_reason",
    [
        "",
        "research_pipeline_execution_retry_configuration_superseded",
        "research_pipeline_execution_retry_path_invalid",
        "research_pipeline_execution_retry_checkpoint_missing",
    ],
)
def test_authority_failures_do_not_preserve_the_approved_checkpoint(
    gate_reason: str,
) -> None:
    assert preserves_approved_execution_checkpoint(gate_reason) is False
