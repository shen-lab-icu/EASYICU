"""Declared cohort columns a host computes from several source concepts.

Owner
-----
The cohort materializer projects one output column from one source concept:
window it, aggregate it, record the transform.  That is the right shape for
almost every column, and the sealed lineage contract enforces it -- a
materialized column carries exactly one source receipt, so it can never claim
provenance it does not have.

Some scientific readings are irreducibly cross-concept.  An
observability-preserving KDIGO stage is the worked example: whether a stay is
"stage 0" or "never assessed" is decided by three evidence receipts together,
and no per-concept aggregation of the stage column alone can tell them apart.

This module is the closed declaration of those derivations: their identifier,
every concept they read, and the exact typed shape of every column they
publish.  It holds no implementation, so both the lineage validator
(:mod:`easyicu.research_agent.intake.materialized_metadata`) and the producer
(:mod:`easyicu.research_agent.cohort.host_derivations`) can read the same
declaration without depending on each other.  A multi-source receipt is
accepted only for a transform declared here, and only when its source set
matches this declaration exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional

from easyicu.concept.metadata_projection import ConceptColumnRole


class HostDerivationError(ValueError):
    """The requested host derivation is not declared, or is declared wrongly."""


@dataclass(frozen=True, slots=True)
class HostDerivedColumn:
    """One output column of a host derivation, with its typed projection."""

    column: str
    role: ConceptColumnRole
    transform_id: str
    #: The source whose unit, bounds and lineage this column inherits.  It must
    #: be in the same quantity space as the output; the remaining source
    #: concepts are recorded in ``derived_from_concepts``.
    primary_concept: str
    aggregation: Optional[str] = None
    #: Whether the column carries the cohort window as its derivation window.
    #: A derivation that summarizes a window must say which window it read.
    windowed: bool = True
    #: Whether a plan may name this column as a design variable.  A derivation
    #: also publishes receipts -- how many rows it read, whether it saw a
    #: complete negative -- which belong in the artifact but are not offered as
    #: exposures or covariates.
    selectable: bool = True
    #: Reader-facing one-liner for the planning menu.  Empty falls back to the
    #: derivation's own summary.
    description: str = ""


@dataclass(frozen=True, slots=True)
class HostDerivation:
    """One closed cross-concept derivation the cohort materializer may run."""

    derivation_id: str
    source_concepts: tuple[str, ...]
    outputs: tuple[HostDerivedColumn, ...]
    summary: str

    def __post_init__(self) -> None:
        if not self.source_concepts or len(set(self.source_concepts)) != len(
            self.source_concepts
        ):
            raise HostDerivationError(
                f"{self.derivation_id}: source concepts must be unique and non-empty"
            )
        if not self.outputs:
            raise HostDerivationError(f"{self.derivation_id}: declares no outputs")
        declared = set(self.source_concepts)
        for output in self.outputs:
            if output.primary_concept not in declared:
                raise HostDerivationError(
                    f"{self.derivation_id}: {output.column!r} names a primary "
                    "concept the derivation does not read"
                )

    @property
    def output_columns(self) -> tuple[str, ...]:
        return tuple(output.column for output in self.outputs)


STRICT_KDIGO_DERIVATION_ID = "strict_kdigo_stage"

_STRICT_KDIGO = HostDerivation(
    derivation_id=STRICT_KDIGO_DERIVATION_ID,
    source_concepts=(
        "aki_stage_creat_reference",
        "aki_stage_uo_reference",
        "aki_stage_rrt_reference",
        "creatinine_evidence_status",
        "urine_evidence_status",
        "rrt_evidence_status",
    ),
    outputs=(
        # A stage, in the same 0-3 space as the component it inherits from.
        # It is deliberately not declared as an aggregation: the window rule is
        # "any positive wins, stage 0 needs an observed complete negative",
        # which no numeric aggregation of one column expresses.
        HostDerivedColumn(
            column="aki_stage_strict",
            role=ConceptColumnRole.VALUE,
            transform_id="strict_kdigo_window_stage",
            primary_concept="aki_stage_creat_reference",
            description=(
                "KDIGO stage 0-3 over the cohort window with an explicit "
                "unknown: stage 0 requires evidence that every component was "
                "observed and negative, so never-assessed stays stay missing "
                "instead of joining the reference group"
            ),
        ),
        # The categorical reading that says why a stage is missing.
        HostDerivedColumn(
            column="aki_ascertainment",
            role=ConceptColumnRole.VALUE,
            transform_id="strict_kdigo_window_ascertainment",
            primary_concept="creatinine_evidence_status",
            description=(
                "why aki_stage_strict is known or missing for a stay: "
                "positive / positive_stage_unresolved / negative_complete / "
                "partial_no_observed_positive / indeterminate"
            ),
        ),
        # Receipts: they travel with the cohort so a reader can audit the
        # reading, but a plan does not name them as design variables.
        HostDerivedColumn(
            column="kidney_complete_negative_observed",
            role=ConceptColumnRole.MEASUREMENT_STATUS,
            transform_id="strict_kdigo_window_complete_negative",
            primary_concept="creatinine_evidence_status",
            selectable=False,
        ),
        HostDerivedColumn(
            column="kidney_window_row_count",
            role=ConceptColumnRole.COUNT,
            transform_id="strict_kdigo_window_row_count",
            primary_concept="creatinine_evidence_status",
            selectable=False,
        ),
    ),
    summary=(
        "Observability-preserving KDIGO stage over the cohort window: a "
        "positive component establishes the stage, stage 0 requires an "
        "observed complete negative, everything else stays unknown."
    ),
)

HOST_DERIVATIONS: Mapping[str, HostDerivation] = MappingProxyType(
    {_STRICT_KDIGO.derivation_id: _STRICT_KDIGO}
)

_BY_TRANSFORM: Mapping[str, tuple[HostDerivation, HostDerivedColumn]] = (
    MappingProxyType(
        {
            output.transform_id: (derivation, output)
            for derivation in HOST_DERIVATIONS.values()
            for output in derivation.outputs
        }
    )
)

_BY_COLUMN: Mapping[str, tuple[HostDerivation, HostDerivedColumn]] = MappingProxyType(
    {
        output.column: (derivation, output)
        for derivation in HOST_DERIVATIONS.values()
        for output in derivation.outputs
    }
)


def host_derivation(derivation_id: str) -> HostDerivation:
    """Resolve one declared derivation, or fail closed."""

    declared = HOST_DERIVATIONS.get(str(derivation_id).strip())
    if declared is None:
        known = ", ".join(sorted(HOST_DERIVATIONS)) or "none"
        raise HostDerivationError(
            f"undeclared host derivation {derivation_id!r}; declared: {known}"
        )
    return declared


def host_derived_transform(
    transform_id: object,
) -> Optional[tuple[HostDerivation, HostDerivedColumn]]:
    """Return the derivation and column a transform id belongs to, if declared."""

    if not isinstance(transform_id, str):
        return None
    return _BY_TRANSFORM.get(transform_id.strip())


def host_derivation_producing(
    column: object,
) -> Optional[tuple[HostDerivation, HostDerivedColumn]]:
    """Return the derivation that publishes ``column``, if any is declared."""

    if not isinstance(column, str):
        return None
    return _BY_COLUMN.get(column.strip())


__all__ = [
    "HOST_DERIVATIONS",
    "STRICT_KDIGO_DERIVATION_ID",
    "HostDerivation",
    "HostDerivationError",
    "HostDerivedColumn",
    "host_derivation",
    "host_derivation_producing",
    "host_derived_transform",
]
