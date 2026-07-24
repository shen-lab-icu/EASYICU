"""Case-neutral names shared by bounded-metric consumers."""

from __future__ import annotations

import re
from typing import Any

_SCALE_DESCRIPTOR_NAMES = {
    "effect_measure",
    "estimand",
    "measure",
    "measure_type",
    "metric",
    "metric_name",
    "scale",
    "statistic",
    "type",
    "unit",
    "units",
}


def normalize_metric_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def is_scale_descriptor_field(value: Any) -> bool:
    name = normalize_metric_key(value)
    return name in _SCALE_DESCRIPTOR_NAMES or name.endswith(
        ("_measure", "_metric", "_scale", "_type", "_unit", "_units")
    )
