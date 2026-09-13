"""The execution-time dependency gate: who may no longer run, and why.

Covers the two explicit producer edges the gate honors — the structural
figure→parent edge and Planner-declared typed ``kind:product`` inputs — plus
the negative cases that must NOT become dependencies (untyped variable names,
clean producers, producers with no record) and transitive skipping after a
dependent step is already marked.
"""

from __future__ import annotations

from threading import RLock

from easyicu.research_agent.execution.phase_support import (
    _step_failed_dependency_record,
)
from easyicu.research_agent.schema import AnalysisStep


def _step(
    step_id: str,
    *,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    intent: str | None = None,
) -> AnalysisStep:
    return AnalysisStep(
        step_id=step_id,
        intent=intent or f"execute {step_id}",
        inputs=list(inputs or ()),
        expected_outputs=list(outputs or ()),
    )


def _record(step_id: str, status: str) -> dict[str, str]:
    return {"step_id": step_id, "status": status}


def _supplier(mapping):
    return lambda: mapping


def test_typed_input_is_skipped_after_its_producer_fails() -> None:
    consumer = _step("05_summary", inputs=["table:04_result"])
    supplier = _supplier({"04": ("table:04_result",)})

    record = _step_failed_dependency_record(
        consumer,
        per_step_records=[_record("04", "execution_failed")],
        shared_lock=RLock(),
        step_outputs_supplier=supplier,
    )

    assert record is not None
    assert record["step_id"] == "04"
    assert record["status"] == "execution_failed"
    assert record["dependency_match"] == "typed_product"


def test_untyped_variable_input_is_not_a_dependency() -> None:
    consumer = _step("05_summary", inputs=["hr"])
    supplier = _supplier({"04": ("hr",)})

    record = _step_failed_dependency_record(
        consumer,
        per_step_records=[_record("04", "execution_failed")],
        shared_lock=RLock(),
        step_outputs_supplier=supplier,
    )

    assert record is None


def test_clean_producer_is_not_a_dependency() -> None:
    consumer = _step("05_summary", inputs=["table:04_result"])
    supplier = _supplier({"04": ("table:04_result",)})

    record = _step_failed_dependency_record(
        consumer,
        per_step_records=[_record("04", "ok")],
        shared_lock=RLock(),
        step_outputs_supplier=supplier,
    )

    assert record is None


def test_producer_without_a_record_is_not_a_dependency() -> None:
    consumer = _step("05_summary", inputs=["statistic:04_estimate"])
    supplier = _supplier({"04": ("statistic:04_estimate",)})

    record = _step_failed_dependency_record(
        consumer,
        per_step_records=[],
        shared_lock=RLock(),
        step_outputs_supplier=supplier,
    )

    assert record is None


def test_skipped_producer_blocks_its_own_consumers() -> None:
    """A step skipped after its producer failed still blocks its consumers."""

    consumer = _step("06_render", inputs=["figure:05_plot"])
    supplier = _supplier({"05": ("figure:05_plot",)})

    record = _step_failed_dependency_record(
        consumer,
        per_step_records=[_record("05", "skipped_dependency_failed")],
        shared_lock=RLock(),
        step_outputs_supplier=supplier,
    )

    assert record is not None
    assert record["step_id"] == "05"
    assert record["status"] == "skipped_dependency_failed"


def test_figure_parent_edge_still_wins_without_typed_inputs() -> None:
    consumer = _step(
        "01_model_training_figure",
        intent=(
            "Render the publication figure(s) declared by step "
            "'01_model_training'."
        ),
    )

    record = _step_failed_dependency_record(
        consumer,
        per_step_records=[_record("01_model_training", "execution_failed")],
        shared_lock=RLock(),
    )

    assert record is not None
    assert record["step_id"] == "01_model_training"
    assert record["status"] == "execution_failed"


def test_figure_parent_with_no_record_is_not_a_dependency() -> None:
    consumer = _step("01_model_training_figure")

    record = _step_failed_dependency_record(
        consumer,
        per_step_records=[],
        shared_lock=RLock(),
    )

    assert record is None


def test_a_steps_own_stale_record_cannot_skip_itself() -> None:
    """A retry must not be blocked by the record the last attempt left.

    When a step both produces and consumes the same typed product (a self
    edge, however unusual), the producer scan would otherwise find its own
    failed record and skip the step before its retry even starts; a step is
    never its own dependency.
    """

    consumer = _step(
        "03_iterate",
        inputs=["table:03_partial"],
        outputs=["table:03_partial"],
    )
    supplier = _supplier({"03_iterate": ("table:03_partial",)})

    record = _step_failed_dependency_record(
        consumer,
        per_step_records=[_record("03_iterate", "execution_failed")],
        shared_lock=RLock(),
        step_outputs_supplier=supplier,
    )

    assert record is None


def test_another_failed_producer_still_blocks_despite_self_edge() -> None:
    """The self-exclusion must not hide a genuinely failed other producer."""

    consumer = _step(
        "03_iterate",
        inputs=["table:03_partial", "table:02_basis"],
        outputs=["table:03_partial"],
    )
    supplier = _supplier(
        {
            "03_iterate": ("table:03_partial",),
            "02_basis": ("table:02_basis",),
        }
    )

    record = _step_failed_dependency_record(
        consumer,
        per_step_records=[
            _record("03_iterate", "execution_failed"),
            _record("02_basis", "execution_failed"),
        ],
        shared_lock=RLock(),
        step_outputs_supplier=supplier,
    )

    assert record is not None
    assert record["step_id"] == "02_basis"
