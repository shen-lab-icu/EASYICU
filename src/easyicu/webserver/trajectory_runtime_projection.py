"""Compile a reviewed Web trajectory design into sealed trajectory authority.

Owner
-----
This module owns the Web-to-Research-Agent projection for fixed-window
longitudinal trajectory clustering.  It is the sibling of
``scientific_runtime_projection`` on a different axis: the current-case
adapters seal one exposure-outcome contract, while the trajectory owners are
carried by ``PipelineConfig.trajectory_scientific_runtime_authority`` and are
outcome-blind by protocol.  The two never compile each other's design.

The reviewable design lives in the StudyContext field ``trajectory_design`` and
is validated by the dependency-neutral contract; this adapter only answers
whether the selected source and materialization can execute it, and refuses by
name when they cannot.  It reads the long panel's *provenance*, never a patient
row, and it never invents a coordinate, a window or a cluster grid: an
undeclared design is not a trajectory study, and a declared design that cannot
execute is a blocker rather than a substituted default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from easyicu.research_agent.contracts.trajectory_design import (
    FixedWindowTrajectoryDesign,
    TrajectoryDesignError,
    load_trajectory_design,
    sealed_trajectory_authority_body,
)
from easyicu.research_agent.trajectory.scientific_runtime_authority import (
    build_trajectory_scientific_runtime_authority,
)

from .scientific_runtime_projection import (
    WebScientificRuntimeProjection,
    WebScientificRuntimeProjectionError,
    signed_projection,
)

__all__ = [
    "TRAJECTORY_ANALYSIS_FAMILY",
    "compile_web_trajectory_runtime_projection",
    "trajectory_family_declared",
    "trajectory_provenance_path",
    "validate_trajectory_design_declaration",
]

TRAJECTORY_ANALYSIS_FAMILY = "trajectory_clustering"
_SUPPORTED_ANALYSIS_UNIT = "icu_stay"
_SUPPORTED_VARIANCE_ESTIMATOR = "model_based"
# The sealed representation owner defines eligibility as a count of windows
# with owner-available SOFA-2 evidence, and reports exclusions under that
# name. A design with no SOFA-2 coordinate cannot state eligibility at all.
_ELIGIBILITY_COORDINATE_PREFIX = "sofa2"


def _design_field(study: Mapping[str, Any]) -> Mapping[str, Any]:
    declared = study.get("trajectory_design")
    return declared if isinstance(declared, Mapping) else {}


def _analysis_design(study: Mapping[str, Any]) -> Mapping[str, Any]:
    design = study.get("analysis_design")
    return design if isinstance(design, Mapping) else {}


def trajectory_family_declared(study: Mapping[str, Any]) -> bool:
    """Whether the reviewed study declares a longitudinal trajectory family."""

    family = str(_analysis_design(study).get("analysis_family") or "").strip()
    return family == TRAJECTORY_ANALYSIS_FAMILY


def trajectory_provenance_path(universe_path: Path) -> Path:
    """The long panel's provenance sidecar beside a materialized universe."""

    return universe_path.with_name(
        f"{universe_path.stem}_trajectory_provenance.json"
    )


def _fail(code: str, message: str, **details: Any) -> None:
    raise WebScientificRuntimeProjectionError(code, message, details=details)


def _typed_design(study: Mapping[str, Any]) -> FixedWindowTrajectoryDesign:
    try:
        design = load_trajectory_design(_design_field(study))
    except TrajectoryDesignError as exc:
        _fail(
            "web_trajectory_design_invalid",
            str(exc),
            field=exc.field,
            design_error_code=exc.code,
        )
        raise  # pragma: no cover - _fail always raises
    if design is None:  # pragma: no cover - guarded by the caller
        raise AssertionError("a declared trajectory design cannot be empty")
    return design


def _require_supported_inference(study: Mapping[str, Any]) -> None:
    design = _analysis_design(study)
    analysis_unit = str(design.get("analysis_unit") or "").strip()
    variance_estimator = str(design.get("variance_estimator") or "").strip()
    if (
        analysis_unit != _SUPPORTED_ANALYSIS_UNIT
        or variance_estimator != _SUPPORTED_VARIANCE_ESTIMATOR
    ):
        _fail(
            "web_trajectory_design_unsupported",
            (
                "The signed trajectory owners cluster one row per ICU stay from "
                "a stay-anchored longitudinal panel. A patient-clustered "
                "materialization cannot carry that panel without dropping the "
                "identity the panel is keyed by, so a cluster-robust or "
                "counts-only ceiling is refused here rather than silently "
                "clustering a different row space."
            ),
            analysis_unit=analysis_unit or None,
            variance_estimator=variance_estimator or None,
            supported_analysis_unit=_SUPPORTED_ANALYSIS_UNIT,
            supported_variance_estimator=_SUPPORTED_VARIANCE_ESTIMATOR,
        )


def _panel_provenance(universe_path: Path) -> Mapping[str, Any]:
    path = trajectory_provenance_path(universe_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _fail(
            "web_trajectory_longitudinal_panel_missing",
            (
                "A trajectory design requires the long per-timepoint panel "
                "beside the materialized universe; this run has none, so the "
                "signed representation owner has nothing to read."
            ),
            expected_artifact=path.name,
            cause=type(exc).__name__,
        )
        raise  # pragma: no cover - _fail always raises
    if not isinstance(payload, Mapping):
        _fail(
            "web_trajectory_longitudinal_panel_missing",
            "The long panel provenance is not a typed object.",
            expected_artifact=path.name,
        )
    return payload


def _require_materialized_window(
    design: FixedWindowTrajectoryDesign, provenance: Mapping[str, Any]
) -> None:
    window = provenance.get("window")
    if not isinstance(window, Sequence) or isinstance(window, (str, bytes)):
        return
    bounds = [value for value in window if isinstance(value, (int, float))]
    if len(bounds) != 2:
        return
    start, end = float(bounds[0]), float(bounds[1])
    if design.window_start_hours < start or design.window_end_hours > end:
        _fail(
            "web_trajectory_window_outside_materialization",
            (
                "The reviewed trajectory window reaches outside the window the "
                "panel was materialized over. Widening it here would change the "
                "user-reviewed materialization scope without a second review."
            ),
            trajectory_window_hours=[
                design.window_start_hours,
                design.window_end_hours,
            ],
            materialized_window_hours=[start, end],
        )


def _require_materialized_concepts(
    design: FixedWindowTrajectoryDesign, provenance: Mapping[str, Any]
) -> None:
    def names(key: str) -> tuple[str, ...]:
        raw = provenance.get(key)
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return ()
        return tuple(str(item) for item in raw)

    materialized = set(names("trajectory_concepts_materialized"))
    missing = [
        concept
        for concept in design.required_concepts
        if concept not in materialized
    ]
    if missing:
        _fail(
            "web_trajectory_concepts_unavailable",
            (
                "Every declared trajectory coordinate must be present in the "
                "materialized longitudinal panel; a coordinate the panel never "
                "observed cannot be scaled, fitted, or reported as missing by "
                "design."
            ),
            missing_concepts=sorted(missing),
            materialized_concepts=sorted(materialized),
            available_unobserved_concepts=sorted(
                names("available_unobserved_concepts")
            ),
            unavailable_concepts=sorted(names("unavailable_concepts")),
        )


def validate_trajectory_design_declaration(
    study: Mapping[str, Any],
) -> FixedWindowTrajectoryDesign | None:
    """The source-independent half of this owner's contract.

    The launch gate calls this before any materialization is spent, and the
    projection calls it again before reading the panel, so a reviewer is told
    about a contradictory or unexecutable declaration at the earliest moment it
    can be known -- without a second copy of the policy.
    """

    declared_design = _design_field(study)
    declared_family = trajectory_family_declared(study)
    if not declared_design and not declared_family:
        return None
    if declared_family and not declared_design:
        _fail(
            "web_trajectory_design_required",
            (
                "A longitudinal trajectory family needs its reviewed design "
                "before launch: which coordinates are modelled, over which "
                "fixed window and grid, which cluster counts are admissible, "
                "and what counts as a stable solution. The host does not "
                "choose those for the study."
            ),
            field="trajectory_design",
            required_fields=["coordinate_concepts"],
            analysis_family=TRAJECTORY_ANALYSIS_FAMILY,
        )
    if declared_design and not declared_family:
        _fail(
            "web_trajectory_family_mismatch",
            (
                "A trajectory design is declared, but the typed analysis family "
                "is not trajectory clustering. The two must agree before the "
                "signed owners can claim the plan."
            ),
            field="analysis_design.analysis_family",
            declared_analysis_family=(
                str(_analysis_design(study).get("analysis_family") or "") or None
            ),
            required_analysis_family=TRAJECTORY_ANALYSIS_FAMILY,
        )

    design = _typed_design(study)
    _require_supported_inference(study)
    if not any(
        concept.startswith(_ELIGIBILITY_COORDINATE_PREFIX)
        for concept in design.coordinate_concepts
    ):
        _fail(
            "web_trajectory_eligibility_coordinate_missing",
            (
                "The signed representation owner counts owner-available SOFA-2 "
                "windows to decide who is eligible for clustering, and reports "
                "every exclusion under that rule. A design whose coordinates "
                "contain no SOFA-2 component would silently exclude the whole "
                "cohort instead of measuring it."
            ),
            coordinate_concepts=list(design.coordinate_concepts),
            required_coordinate_prefix=_ELIGIBILITY_COORDINATE_PREFIX,
        )
    return design


def compile_web_trajectory_runtime_projection(
    *,
    study: Mapping[str, Any],
    universe_path: Path,
    scientific_configuration_sha256: str,
) -> WebScientificRuntimeProjection | None:
    """Seal the reviewed trajectory design, or refuse it by name.

    Returns ``None`` when the study is not a trajectory study at all, so the
    ordinary association and survival routes are untouched.
    """

    design = validate_trajectory_design_declaration(study)
    if design is None:
        return None
    provenance = _panel_provenance(universe_path)
    _require_materialized_window(design, provenance)
    _require_materialized_concepts(design, provenance)

    authority = build_trajectory_scientific_runtime_authority(
        sealed_trajectory_authority_body(
            design,
            protocol_content_sha256=scientific_configuration_sha256,
        )
    )
    return signed_projection(
        authority.model_dump(mode="json"),
        scientific_configuration_sha256=scientific_configuration_sha256,
    )
