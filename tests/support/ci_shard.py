"""Split one full CI run across several machines by test file.

``-p tests.support.ci_shard --ci-shard=K/N`` keeps the files assigned to
shard K of N and deselects the rest. Files are assigned greedily, largest
first, to the shard with the fewest collected tests, so every worker and every
shard computes the same split from the same collection. The full collection is
kept on ``config`` for plugins that check completeness (Figure 2 V4 mapping).
Plain pytest does not activate sharding.
"""
from __future__ import annotations

from collections import Counter

import pytest

FULL_COLLECTION_ATTR = "_ci_shard_full_nodeids"


def pytest_addoption(parser):
    group = parser.getgroup("ci-shard")
    group.addoption("--ci-shard", default=None, help="run shard K of N, written K/N (1-based)")


def parse_shard(value: str) -> tuple[int, int]:
    try:
        index, count = (int(part) for part in value.split("/"))
    except ValueError:
        raise pytest.UsageError(f"--ci-shard must be K/N, got {value!r}") from None
    if count < 1 or not 1 <= index <= count:
        raise pytest.UsageError(f"--ci-shard {value!r} is outside 1..N")
    return index, count


def assign_files(sizes: Counter, count: int) -> dict[str, int]:
    """File -> 1-based shard, balancing collected test counts."""
    loads = [0] * count
    assignment = {}
    for name, size in sorted(sizes.items(), key=lambda item: (-item[1], item[0])):
        shard = min(range(count), key=lambda position: (loads[position], position))
        assignment[name] = shard + 1
        loads[shard] += size
    return assignment


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    value = config.getoption("--ci-shard")
    if not value:
        return
    index, count = parse_shard(value)
    setattr(config, FULL_COLLECTION_ATTR, frozenset(item.nodeid for item in items))
    sizes = Counter(item.nodeid.split("::", 1)[0] for item in items)
    assignment = assign_files(sizes, count)
    keep = [item for item in items if assignment[item.nodeid.split("::", 1)[0]] == index]
    drop = [item for item in items if assignment[item.nodeid.split("::", 1)[0]] != index]
    if drop:
        config.hook.pytest_deselected(items=drop)
    items[:] = keep
