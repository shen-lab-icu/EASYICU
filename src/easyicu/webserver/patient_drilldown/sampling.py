"""Bounded trajectory selection that retains endpoints and observed extremes."""

from collections.abc import Sequence


def trajectory_indices(values: Sequence[float], limit: int) -> list[int]:
    """Select ordered observations, preserving first, last, minimum and maximum."""
    count = len(values)
    if count <= limit:
        return list(range(count))
    selected = {
        0,
        count - 1,
        min(range(count), key=values.__getitem__),
        max(range(count), key=values.__getitem__),
    }
    while len(selected) < limit:
        ordered = sorted(selected)
        left, right = max(zip(ordered, ordered[1:]), key=lambda pair: pair[1] - pair[0])
        selected.add((left + right) // 2)
    return sorted(selected)
