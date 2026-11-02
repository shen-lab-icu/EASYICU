"""The CI shard split covers every test file exactly once."""
from __future__ import annotations

from collections import Counter

import pytest

from tests.support.ci_shard import assign_files, parse_shard


def test_every_file_lands_in_exactly_one_balanced_shard() -> None:
    sizes = Counter({f"tests/t{i}.py": size for i, size in enumerate([40, 30, 20, 10, 10, 5, 5, 1])})
    assignment = assign_files(sizes, 3)
    assert set(assignment) == set(sizes)
    assert set(assignment.values()) == {1, 2, 3}
    loads = Counter()
    for name, shard in assignment.items():
        loads[shard] += sizes[name]
    assert max(loads.values()) - min(loads.values()) <= max(sizes.values())
    # Every worker and shard recomputes the same split from the same collection.
    assert assign_files(Counter(dict(reversed(list(sizes.items())))), 3) == assignment


@pytest.mark.parametrize("value", ["0/5", "6/5", "1/0", "x", "1-5"])
def test_malformed_shard_is_refused(value: str) -> None:
    with pytest.raises(pytest.UsageError):
        parse_shard(value)
