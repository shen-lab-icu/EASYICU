"""Pin the diagnostics emitted when synthetic benchmark integrity checks fail."""

from collections import Counter
from types import SimpleNamespace

import pytest

from tests.support import benchmark_integrity as checks


@pytest.mark.parametrize(
    ("expected", "changed", "private_values", "message"),
    [
        pytest.param(
            ("kite-blue", "kite-amber"),
            ("kite-amber", "kite-blue"),
            ("kite-blue", "kite-amber"),
            "Split task IDs differ from the frozen sequence.",
            id="split-order",
        ),
        pytest.param(
            {"orchard-path": 1},
            Counter({"orchard-path": 1, "river-crossing": 1}),
            ("orchard-path", "river-crossing"),
            "Task difficulty coverage differs from the frozen counts.",
            id="difficulty-labels",
        ),
        pytest.param(
            {"garden-catalogue": 1},
            Counter({"seed-library": 1}),
            ("garden-catalogue", "seed-library"),
            "Task database coverage differs from the frozen counts.",
            id="database-labels",
        ),
        pytest.param(
            "a" * 64,
            "Which paper kite has the longest tail?",
            ("Which paper kite has the longest tail?",),
            "Frozen asset digest differs from the manifest.",
            id="malformed-digest",
        ),
    ],
)
def test_exact_comparisons_hide_item_values(
    monkeypatch: pytest.MonkeyPatch,
    expected: object,
    changed: object,
    private_values: tuple[str, ...],
    message: str,
) -> None:
    loaded = SimpleNamespace(value=expected)
    checks.assert_integrity_equal(loaded.value, expected, message)
    monkeypatch.setattr(loaded, "value", changed)

    with pytest.raises((AssertionError, pytest.fail.Exception)) as failure:
        checks.assert_integrity_equal(loaded.value, expected, message)

    failure_text = str(failure.value)
    assert all(value not in failure_text for value in private_values)
    assert failure_text == message
    assert failure.value.pytrace is False


def test_distinct_question_failure_reports_only_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    questions = (
        "Which lantern stays lit after sunset?",
        "Where does the river path end?",
        "When does the seed library open?",
    )
    rows = [{"question": question} for question in questions]
    checks.assert_distinct_question_count((row["question"] for row in rows), 3)
    monkeypatch.setitem(rows[-1], "question", questions[0])

    with pytest.raises((AssertionError, pytest.fail.Exception)) as failure:
        checks.assert_distinct_question_count((row["question"] for row in rows), 3)

    failure_text = str(failure.value)
    assert all(question not in failure_text for question in questions)
    assert failure_text == "Expected 3 distinct questions, observed 2."
    assert failure.value.pytrace is False


@pytest.mark.parametrize(
    ("needle", "original", "changed", "private_values", "message"),
    [
        pytest.param(
            "pottery_glaze_temperature",
            ("pottery_glaze_temperature",),
            ("pottery_glaze_temperature_sensor",),
            ("pottery_glaze_temperature", "pottery_glaze_temperature_sensor"),
            "The omitted concept was not attributed to the affected task.",
            id="concept-membership",
        ),
        pytest.param(
            "duplicate JSON key",
            "duplicate JSON key: task_id",
            "Invalid question: When will the pottery kiln cool?",
            ("When will the pottery kiln cool?",),
            "Taskbank parser did not report a duplicate JSON key.",
            id="parser-detail",
        ),
    ],
)
def test_membership_checks_hide_item_values(
    monkeypatch: pytest.MonkeyPatch,
    needle: str,
    original: tuple[str, ...] | str,
    changed: tuple[str, ...] | str,
    private_values: tuple[str, ...],
    message: str,
) -> None:
    loaded = SimpleNamespace(values=original)
    checks.assert_integrity_contains(needle, loaded.values, message)
    monkeypatch.setattr(loaded, "values", changed)

    with pytest.raises((AssertionError, pytest.fail.Exception)) as failure:
        checks.assert_integrity_contains(needle, loaded.values, message)

    failure_text = str(failure.value)
    assert all(value not in failure_text for value in private_values)
    assert failure_text == message
    assert failure.value.pytrace is False


def test_unexpected_validation_error_hides_input_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item_id = "brass-compass-17"
    question = "How far is the lighthouse from the pier?"
    loaded = SimpleNamespace(detail="")
    monkeypatch.setattr(loaded, "detail", f"Invalid item {item_id}: {question}")

    with pytest.raises((ValueError, pytest.fail.Exception)) as failure:
        with checks.redact_integrity_errors():
            raise ValueError(loaded.detail)

    failure_text = str(failure.value)
    assert all(value not in failure_text for value in (item_id, question))
    assert failure_text == "Benchmark integrity check failed."
    assert failure.value.pytrace is False
    assert failure.value.__suppress_context__ is True
