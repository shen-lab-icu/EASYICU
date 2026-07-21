"""Compact, lossless variable metadata for coder prompt transport.

Wide fixed-window trajectory panels repeat the same family policy on every
physical time-bin column.  The coder still needs every exact column, window,
observed domain, and missingness fact, but it does not need the identical
family policy copied dozens of times.  This module separates those shared and
per-column coordinates without selecting variables or changing scientific
authority.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from ..schema import ConceptDescriptor


@dataclass(frozen=True)
class CompactTrajectoryPromptProjection:
    """Shared family lines and exact per-column lines for repeated windows."""

    shared_lines: tuple[str, ...]
    variable_lines: tuple[tuple[str, str], ...]


def format_observed_domain(domain: Optional[Dict[str, Any]]) -> str:
    """Render cohort-observed values as compact, fact-only prompt metadata."""

    if not domain:
        return ""
    if domain.get("is_constant"):
        return " observed=CONSTANT(single value; no variation to model)"
    if domain.get("is_binary"):
        return (
            " observed={0,1} BINARY(already 2-level; a numeric cutoff >1 is "
            "degenerate)"
        )
    levels = domain.get("levels")
    if levels:
        shown = ",".join(str(value) for value in levels[:6])
        more = "…" if len(levels) > 6 else ""
        return f" observed_levels={{{shown}{more}}}(categorical; encode as-is)"
    lo, hi = domain.get("min"), domain.get("max")
    n_unique = domain.get("n_unique")
    if lo is not None and hi is not None:
        return f" observed=[{lo:g},{hi:g}] n_unique={n_unique}"
    if n_unique is not None:
        return f" observed_n_unique={n_unique}"
    return ""


def _normalised_description(variable: ConceptDescriptor) -> str:
    return " ".join(str(variable.description or "").split())


def _shared_signature(variable: ConceptDescriptor) -> tuple[object, ...]:
    metadata = variable.fixed_window_trajectory
    if metadata is None:  # pragma: no cover - caller filters this condition
        raise ValueError("fixed-window trajectory metadata is required")
    return (
        variable.role.value,
        variable.dtype,
        variable.unit,
        tuple(variable.valid_range or ()),
        variable.source_concept,
        _normalised_description(variable),
        variable.is_ordinal,
        tuple(variable.ordinal_levels or ()),
        (
            variable.aggregation_default.value
            if variable.aggregation_default is not None
            else "unspecified"
        ),
        variable.temporal_resolution,
        metadata.window_width_hours,
        metadata.time_axis,
        metadata.anchor,
        metadata.source_scale,
        metadata.representation_kind,
        metadata.observed_fractional_values,
        tuple(variable.pitfalls),
    )


def _shared_line(
    variable: ConceptDescriptor,
    *,
    group_id: str,
    member_count: int,
    families: tuple[str, ...],
) -> str:
    metadata = variable.fixed_window_trajectory
    if metadata is None:  # pragma: no cover - caller filters this condition
        raise ValueError("fixed-window trajectory metadata is required")
    fields = [
        f"  - trajectory_group={group_id!r}",
        f"members={member_count}",
        f"families={list(families)!r}",
        f"role={variable.role.value}",
        f"dtype={variable.dtype}",
    ]
    if variable.unit:
        fields.append(f"unit={variable.unit}")
    if variable.valid_range:
        fields.append(
            "plausibility_range(flag_only;never_exclude_rows)="
            f"{variable.valid_range}"
        )
    if variable.source_concept:
        fields.append(f"source_concept={variable.source_concept}")
    description = _normalised_description(variable)
    if description:
        fields.append(f"description={description!r}")
    fields.extend(
        [
            f"is_ordinal={str(variable.is_ordinal).lower()}",
            f"ordinal_levels={variable.ordinal_levels or 'unspecified'}",
            "agg_default="
            + (
                variable.aggregation_default.value
                if variable.aggregation_default is not None
                else "unspecified"
            ),
            f"window_width={metadata.window_width_hours:g}h",
            f"time_axis={metadata.time_axis}",
            f"source_scale={metadata.source_scale}",
            f"representation={metadata.representation_kind}",
            f"anchor={metadata.anchor or 'unspecified_agent_must_declare'}",
            "observed_fractional_values="
            f"{str(metadata.observed_fractional_values).lower()}",
        ]
    )
    if variable.temporal_resolution:
        fields.append(f"temporal_resolution={variable.temporal_resolution!r}")
    if variable.pitfalls:
        fields.append(f"pitfalls={variable.pitfalls!r}")
    return " | ".join(fields)


def _variable_line(variable: ConceptDescriptor, *, group_id: str) -> str:
    metadata = variable.fixed_window_trajectory
    if metadata is None:  # pragma: no cover - caller filters this condition
        raise ValueError("fixed-window trajectory metadata is required")
    missingness = ""
    if variable.missingness is not None:
        missingness = (
            f" m={variable.missingness.fraction_missing:.1%}/"
            f"{variable.missingness.missingness_severity}"
        )
    domain = variable.observed_domain or {}
    if domain.get("is_constant"):
        observed = " obs=constant(no-model-variation)"
    elif domain.get("is_binary"):
        observed = " obs=binary{0,1}(already-two-level)"
    elif domain.get("levels"):
        levels = domain["levels"]
        shown = ",".join(str(value) for value in levels[:6])
        observed = f" obs=levels{{{shown}{'…' if len(levels) > 6 else ''}}}"
    elif domain.get("min") is not None and domain.get("max") is not None:
        observed = (
            f" obs={domain['min']:g}:{domain['max']:g}" f"/u{domain.get('n_unique')}"
        )
    elif domain.get("n_unique") is not None:
        observed = f" obs=u{domain['n_unique']}"
    else:
        observed = ""
    return (
        f"- {variable.name} | g={group_id!r}"
        f" f={metadata.family!r}"
        f" t=[{metadata.window_start_hours:g},{metadata.window_end_hours:g})h"
        f"{observed}{missingness}"
    )


def compact_fixed_window_trajectory_prompt(
    variables: Sequence[ConceptDescriptor],
    *,
    minimum_group_size: int = 4,
) -> CompactTrajectoryPromptProjection:
    """Factor exact repeated window metadata without dropping any column.

    Only columns with an identical shared-policy signature are grouped.  Small
    groups retain the ordinary variable formatter, avoiding a second notation
    when it would not materially reduce transport.
    """

    grouped: dict[tuple[object, ...], list[ConceptDescriptor]] = defaultdict(list)
    for variable in variables:
        if variable.fixed_window_trajectory is not None:
            grouped[_shared_signature(variable)].append(variable)

    accepted = [
        (signature, members)
        for signature, members in grouped.items()
        if len(members) >= max(2, int(minimum_group_size))
    ]
    shared_lines: list[str] = []
    variable_lines: list[tuple[str, str]] = []
    for group_number, (_signature, members) in enumerate(accepted, start=1):
        families = tuple(
            dict.fromkeys(
                variable.fixed_window_trajectory.family  # type: ignore[union-attr]
                for variable in members
            )
        )
        group_id = f"trajectory_policy#{group_number}"
        shared_lines.append(
            _shared_line(
                members[0],
                group_id=group_id,
                member_count=len(members),
                families=families,
            )
        )
        variable_lines.extend(
            (variable.name, _variable_line(variable, group_id=group_id))
            for variable in members
        )
    return CompactTrajectoryPromptProjection(
        shared_lines=tuple(shared_lines),
        variable_lines=tuple(variable_lines),
    )


__all__ = [
    "CompactTrajectoryPromptProjection",
    "compact_fixed_window_trajectory_prompt",
    "format_observed_domain",
]
