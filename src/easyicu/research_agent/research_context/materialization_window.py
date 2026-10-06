"""The window the Web host materialized a study's feature columns over.

``data_constraints.materialization_window`` is the study's time window as the
launch executed it (role ``outer_observation_window``): every time-varying
feature column was summarized from ICU admission through ``hours``, or
``observation_hours`` when the study binds only that.  It bounds measurement
opportunity; it is not a phenotype's definition window or the outcome's
follow-up.  A context without it (the CLI, a benchmark) records no physical
window: its ``time_windows`` name analysis windows, which default to the
whole stay, not what was materialized.
"""

from __future__ import annotations

import math
from typing import Mapping

from ..schema import ResearchContext
from .concept_population import context_data_constraints
from .temporal_semantics import normalise_time_anchor


def host_materialization_window_hours(context: ResearchContext) -> float | None:
    """Hours after ICU admission the host summarized feature columns through.

    ``None`` when the context records no readable window counted from ICU
    admission.
    """

    window = context_data_constraints(context).get("materialization_window")
    if (
        not isinstance(window, Mapping)
        or window.get("role") != "outer_observation_window"
        or normalise_time_anchor(str(window.get("anchor") or "")) != "icu_admission"
    ):
        return None
    value = window.get("hours")
    if value is None:
        value = window.get("observation_hours")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    hours = float(value)
    return hours if math.isfinite(hours) and hours > 0 else None


__all__ = ["host_materialization_window_hours"]
