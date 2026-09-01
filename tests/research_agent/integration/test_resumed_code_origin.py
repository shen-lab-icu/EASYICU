from __future__ import annotations

import hashlib

import pytest

from easyicu.research_agent.authority.typed_binding import (
    host_authored_generation_mode,
    host_owns_input_binding_receipts,
)
from easyicu.research_agent.execution.phase_support import (
    _root_generation_mode_for_resumed_code,
)


def _digest(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("source_mode", ["fallback", "deterministic_standard"])
def test_repeated_resume_recovers_digest_bound_host_origin(source_mode: str) -> None:
    code = "print('same signed code')\n"
    digest = _digest(code)
    history = [
        {
            "generation_mode": source_mode,
            "executed_code_sha256": digest,
        },
        {
            "generation_mode": "resumed_code_reuse",
            "resumed_from_generation_mode": source_mode,
            "executed_code_sha256": digest,
        },
        {
            "generation_mode": "resumed_code_reuse",
            "resumed_from_generation_mode": "resumed_code_reuse",
            "executed_code_sha256": digest,
        },
    ]

    recovered = _root_generation_mode_for_resumed_code(
        code=code,
        prior_step_record={"generation_mode": "resumed_code_reuse"},
        prior_attempt_records=history,
    )

    assert recovered == source_mode
    assert host_authored_generation_mode(recovered)
    assert host_owns_input_binding_receipts(
        deterministic_standard_executor_used=False,
        deterministic_fallback_used=False,
        sealed_renderer_repair=False,
        resumed_from_generation_mode=recovered,
    )


def test_digest_bound_generated_code_stays_generated() -> None:
    code = "print('agent code')\n"
    recovered = _root_generation_mode_for_resumed_code(
        code=code,
        prior_step_record={"generation_mode": "resumed_code_reuse"},
        prior_attempt_records=[
            {
                "generation_mode": "llm",
                "concept_approved_code_sha256": _digest(code),
            }
        ],
    )

    assert recovered == "llm"
    assert not host_authored_generation_mode(recovered)
    assert not host_owns_input_binding_receipts(
        deterministic_standard_executor_used=False,
        deterministic_fallback_used=False,
        sealed_renderer_repair=False,
        resumed_from_generation_mode=recovered,
    )


def test_unrelated_history_cannot_reclassify_selected_code() -> None:
    recovered = _root_generation_mode_for_resumed_code(
        code="print('selected')\n",
        prior_step_record={"generation_mode": "resumed_code_reuse"},
        prior_attempt_records=[
            {
                "generation_mode": "fallback",
                "executed_code_sha256": _digest("print('different')\n"),
            }
        ],
    )

    assert recovered == "capsule"
