"""Bind a study-owned repeated-unit design to executable model contracts.

The compatibility input currently lives inside ``data_constraints`` as a JSON
object.  Only the exact ``analysis_design`` object is parsed; ordinary prose,
including mentions or negations of clustering, has no authority.  The output
is the typed plan contract consumed unchanged by execution and scientific
review.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ..contracts.dependence import PlannedDependenceRequirement
from ..schema import AnalysisPlan, ResearchContext


class DependenceAuthorityError(ValueError):
    """A declared dependence design conflicts with its owner-issued authority."""


class _AnalysisDesign(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Sibling coordinate owned by ``analysis_types``.  The Web StudyContext
    # transports the scientific family and repeated-unit design in the same
    # closed ``analysis_design`` envelope.  Dependence authority must preserve
    # that known coordinate without interpreting it; family validation remains
    # with its owner.  All other unknown keys are still rejected below by the
    # model's ``extra='forbid'`` contract.
    analysis_family: str | None = None
    analysis_unit: Literal[
        "row", "icu_stay", "hospital_admission", "patient", "site"
    ]
    cluster_unit: Literal["hospital_admission", "patient", "site", "custom"] | None = (
        None
    )
    variance_estimator: Literal[
        "model_based", "heteroskedasticity_robust", "cluster_robust"
    ]


def _requested_cluster_design(context: ResearchContext) -> _AnalysisDesign | None:
    preferences = context.user_preferences
    raw = getattr(preferences, "data_constraints", None)
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # Free text is intentionally not an authority source.
        return None
    if not isinstance(payload, Mapping):
        return None
    design = payload.get("analysis_design")
    if design is None:
        return None
    if not isinstance(design, Mapping):
        raise DependenceAuthorityError("analysis_design must be a typed object")
    try:
        parsed = _AnalysisDesign.model_validate(dict(design))
    except ValueError as exc:
        raise DependenceAuthorityError(
            "analysis_design does not match the closed repeated-unit contract: "
            + str(exc)
        ) from exc
    if parsed.variance_estimator == "cluster_robust" and parsed.cluster_unit is None:
        raise DependenceAuthorityError(
            "cluster_robust analysis_design requires cluster_unit"
        )
    if parsed.variance_estimator != "cluster_robust" and parsed.cluster_unit is not None:
        raise DependenceAuthorityError(
            "cluster_unit is valid only for cluster_robust analysis_design"
        )
    if (
        parsed.analysis_unit != "icu_stay"
        or parsed.variance_estimator != "cluster_robust"
        or parsed.cluster_unit != "patient"
    ):
        return None
    return parsed


def _direct_patient_source(context: ResearchContext) -> str | None:
    provenance = context.cohort.provenance or {}
    declared = [
        str(value or "").strip()
        for value in provenance.get("patient_id_columns") or ()
    ]
    available = set(context.cohort.id_columns)
    return next((value for value in declared if value and value in available), None)


def context_patient_group_authority(
    context: ResearchContext,
) -> PlannedDependenceRequirement | None:
    """Compile owner-issued row identity without inferring identifier roles."""

    provenance = context.cohort.provenance or {}
    replacement = provenance.get("replacement_row_identity")
    if isinstance(replacement, Mapping):
        source = str(replacement.get("output_identity_column") or "").strip()
        derivation = replacement.get("patient_group_derivation")
        mapping_sha = str(replacement.get("mapping_file_sha256") or "")
        if (
            source in context.cohort.id_columns
            and isinstance(derivation, Mapping)
            and derivation.get("algorithm") == "prefix_before_:s"
            and derivation.get("delimiter") == ":s"
            and len(mapping_sha) == 64
            and all(char in "0123456789abcdef" for char in mapping_sha.casefold())
        ):
            return PlannedDependenceRequirement(
                group_source=source,
                group_derivation="prefix_before_delimiter",
                delimiter=":s",
            )
    source = _direct_patient_source(context)
    if source:
        return PlannedDependenceRequirement(
            group_source=source,
            group_derivation="identity",
        )
    return None


def context_dependence_authority(
    context: ResearchContext,
) -> PlannedDependenceRequirement | None:
    """Compile the requested design plus exact grouping authority, or no authority."""

    if _requested_cluster_design(context) is None:
        return None
    return context_patient_group_authority(context)


def dependence_matches_context(
    *,
    context: ResearchContext,
    dependence: PlannedDependenceRequirement | None,
) -> bool:
    authority = context_dependence_authority(context)
    return bool(authority is not None and dependence == authority)


def bind_context_dependence_authority(
    *,
    plan: AnalysisPlan,
    context: ResearchContext,
) -> AnalysisPlan:
    """Attach one immutable authority to every adjusted-association requirement.

    Existing declarations are accepted only when byte-equivalent after typed
    normalization.  A Planner cannot invent a grouping column, and a stale plan
    cannot keep a contract after the StudyContext removes its authority.
    """

    authority = context_dependence_authority(context)
    existing = [
        requirement.dependence
        for step in plan.steps
        for requirement in step.model_requirements
        if requirement.dependence is not None
    ]
    existing.extend(
        spec.dependence
        for step in plan.steps
        if (spec := step.exposure_outcome_distribution_spec) is not None
        and spec.dependence is not None
    )
    if authority is None:
        if existing:
            raise DependenceAuthorityError(
                "analysis dependence was declared without a matching "
                "StudyContext authority"
            )
        return plan
    if any(item != authority for item in existing):
        raise DependenceAuthorityError(
            "analysis dependence conflicts with the StudyContext grouping authority"
        )

    changed = False
    steps = []
    for step in plan.steps:
        requirements = []
        for requirement in step.model_requirements:
            if requirement.dependence is None:
                requirement = requirement.model_copy(update={"dependence": authority})
                changed = True
            requirements.append(requirement)
        update: dict[str, object] = {}
        if requirements != list(step.model_requirements):
            update["model_requirements"] = requirements
        distribution = step.exposure_outcome_distribution_spec
        if (
            distribution is not None
            and distribution.dependence is None
        ):
            update["exposure_outcome_distribution_spec"] = distribution.model_copy(
                update={"dependence": authority}
            )
            changed = True
        steps.append(step.model_copy(update=update) if update else step)
    return plan.model_copy(update={"steps": steps}) if changed else plan


__all__ = [
    "DependenceAuthorityError",
    "bind_context_dependence_authority",
    "context_dependence_authority",
    "context_patient_group_authority",
    "dependence_matches_context",
]
