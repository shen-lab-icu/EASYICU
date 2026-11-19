"""Assertions shared by benchmark integrity checks and synthetic governance tests."""

from collections.abc import Collection, Iterable, Iterator
from contextlib import contextmanager

import pytest


@contextmanager
def redact_integrity_errors() -> Iterator[None]:
    """Suppress input details in unexpected parsing and validation failures."""
    try:
        yield
    except Exception:
        raise pytest.fail.Exception(
            "Benchmark integrity check failed.", pytrace=False
        ) from None


def assert_integrity_equal(actual: object, expected: object, message: str) -> None:
    if actual != expected:
        pytest.fail(message, pytrace=False)


def assert_integrity_contains(
    needle: str, values: Collection[str], message: str
) -> None:
    if needle not in values:
        pytest.fail(message, pytrace=False)


def assert_distinct_question_count(questions: Iterable[str], expected: int) -> None:
    observed = len(set(questions))
    if observed != expected:
        pytest.fail(
            f"Expected {expected} distinct questions, observed {observed}.",
            pytrace=False,
        )
