from __future__ import annotations

from dataclasses import replace

import pytest

from easyicu.research_agent.execution.step_executor_registry import (
    AmbiguousExecutorOwnership,
    StepExecutor,
    StepExecutorContext,
    StepExecutorRegistry,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _context() -> StepExecutorContext:
    step = AnalysisStep(
        step_id="01_table",
        intent="Build the declared table",
        method="descriptive",
        inputs=[],
        expected_outputs=["table:summary"],
    )
    return StepExecutorContext(
        step=step,
        plan=AnalysisPlan(research_question="Describe the cohort", steps=[step]),
    )


def _executor(key: str, owns: bool = True) -> StepExecutor:
    return StepExecutor(
        key=key,
        owns=lambda _context: owns,
        render=lambda _context: "print('owned')",
        analysis_kind=key,
        selection_reason=f"{key}_contract",
        progress_message=f"Using {key}",
        consumed_input_keys=lambda _context: (),
    )


def test_registry_calls_every_owner_through_one_context_to_selection_interface() -> (
    None
):
    registry = StepExecutorRegistry()
    registry.declare(_executor("first", owns=False))
    registry.declare(_executor("second"))
    trace = []

    selected = registry.select(_context(), trace=trace)

    assert selected is not None
    assert selected.analysis_kind == "second"
    assert selected.code == "print('owned')"
    assert [(row.analysis_kind, row.outcome) for row in trace] == [
        ("first", "contract_declined"),
        ("second", "selected"),
    ]


def test_registry_refuses_duplicate_executor_key() -> None:
    registry = StepExecutorRegistry()
    registry.declare(_executor("summary"))

    with pytest.raises(ValueError, match="already declared: summary"):
        registry.declare(_executor("summary"))


def test_registry_refuses_an_empty_executor_key() -> None:
    registry = StepExecutorRegistry()

    with pytest.raises(ValueError, match="key is required"):
        registry.declare(_executor("  "))


@pytest.mark.parametrize("keys", [("alpha", "beta"), ("beta", "alpha")])
def test_semantic_overlap_refuses_all_rendering_regardless_of_registration_order(keys):
    events = []
    registry = StepExecutorRegistry()
    for key in keys:
        registry.declare(
            replace(
                _executor(key),
                owns=lambda _c, key=key: events.append(f"claim:{key}") or True,
                render=lambda _c, key=key: events.append(f"render:{key}") or "code",
            )
        )
    trace = []
    with pytest.raises(AmbiguousExecutorOwnership) as caught:
        registry.select(_context(), trace=trace)
    assert caught.value.code == "ambiguous_executor_ownership"
    assert caught.value.step_id == "01_table"
    assert caught.value.owner_keys == ("alpha", "beta")
    assert events == [f"claim:{key}" for key in keys]
    assert [row.outcome for row in trace] == ["ambiguous_ownership"] * 2


def test_all_claims_finish_before_unique_owner_renders():
    events = []
    registry = StepExecutorRegistry()
    registry.declare(
        replace(
            _executor("owner"),
            owns=lambda _c: events.append("owner_claim") or True,
            render=lambda _c: events.append("render") or "code",
        )
    )
    registry.declare(
        replace(
            _executor("decliner"),
            owns=lambda _c: events.append("declined_claim") or False,
        )
    )
    assert registry.select(_context()).code == "code"
    assert events == ["owner_claim", "declined_claim", "render"]


def test_inapplicable_route_cannot_claim_or_render():
    registry = StepExecutorRegistry()
    registry.declare(
        replace(
            _executor("inapplicable"),
            applicable=lambda _c: False,
            owns=lambda _c: pytest.fail("inapplicable owner queried"),
            render=lambda _c: pytest.fail("inapplicable owner rendered"),
        )
    )
    registry.declare(_executor("unique"))
    assert registry.select(_context()).analysis_kind == "unique"


def test_later_owner_query_failure_cannot_render_an_earlier_claim():
    registry = StepExecutorRegistry()
    registry.declare(
        replace(_executor("first"), render=lambda _c: pytest.fail("rendered"))
    )

    def failed_query(_context):
        raise ValueError("owner_contract_invalid")

    registry.declare(replace(_executor("second"), owns=failed_query))
    with pytest.raises(ValueError, match="owner_contract_invalid"):
        registry.select(_context())


def test_zero_owners_preserves_caller_governed_fallback():
    registry = StepExecutorRegistry()
    registry.declare(_executor("declined", owns=False))
    assert registry.select(_context()) is None


def test_receipt_refusal_cannot_hide_a_second_scientific_owner():
    from easyicu.research_agent.authority.plausibility import FlagOnlyPlausibilityScope

    context = replace(
        _context(),
        plausibility_scope=FlagOnlyPlausibilityScope(
            step_id="01_table",
            expected_columns=("age",),
            source_contracts_sha256="a" * 64,
            authority_kind="test_contract",
        ),
    )
    registry = StepExecutorRegistry()
    registry.declare(
        replace(
            _executor("receipt_blocked"),
            blocks_on_plausibility_receipt=True,
            render=lambda _c: pytest.fail("receipt-blocked renderer called"),
        )
    )
    registry.declare(
        replace(
            _executor("other"),
            render=lambda _c: pytest.fail("second renderer called"),
        )
    )
    with pytest.raises(AmbiguousExecutorOwnership):
        registry.select(context)


def test_reused_trace_does_not_rewrite_a_previous_selection():
    registry = StepExecutorRegistry()
    registry.declare(_executor("owner"))
    trace = []
    registry.select(_context(), trace=trace)
    first = trace[0]
    registry.select(_context(), trace=trace)
    assert trace[0] is first
    assert [row.outcome for row in trace] == ["selected", "selected"]
