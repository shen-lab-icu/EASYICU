"""Typed, dependency-neutral scientific model-term declarations.

The Planner owns how each exposure and adjustment variable enters a model.
Executors consume this contract; they must not infer categorical treatment,
level ordering, or reference levels from pandas dtypes or column names.
"""

from __future__ import annotations

from typing import Iterable, List, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


ModelTermRole = Literal["exposure", "covariate"]
ModelTermCoding = Literal[
    "continuous",
    "binary",
    "categorical",
    "ordinal_linear",
]
ModelTermTransform = Literal[
    "identity",
    "treatment_contrast",
    "declared_level_index",
]


def level_spelling(value: object) -> str:
    """Return the one stable spelling used to bind declared and observed levels."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        if value.is_integer():
            return str(int(value))
        return repr(value)
    return str(value).strip()


class ModelTermSpec(BaseModel):
    """Exactly how one source variable enters a statistical model.

    ``continuous`` is numeric identity coding. ``binary`` and ``categorical``
    use treatment contrasts against an explicit reference. ``ordinal_linear``
    maps the declared ordered levels to ``0..k-1`` and therefore estimates one
    coefficient per declared level increment. No coding has an observed-data
    default.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    role: ModelTermRole
    coding: ModelTermCoding
    levels: Optional[List[str]] = None
    reference_level: Optional[str] = None
    transform: ModelTermTransform

    @field_validator("name")
    @classmethod
    def _nonblank_name(cls, value: str) -> str:
        name = str(value or "").strip()
        if not name:
            raise ValueError("model term name must be non-empty")
        return name

    @field_validator("levels")
    @classmethod
    def _closed_unique_levels(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return None
        levels = [level_spelling(item) for item in value]
        if any(not item for item in levels):
            raise ValueError("model term levels must be non-empty")
        if len(levels) != len(set(levels)):
            raise ValueError("model term levels must be unique")
        return levels

    @field_validator("reference_level")
    @classmethod
    def _normalise_reference(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        reference = level_spelling(value)
        if not reference:
            raise ValueError("model term reference_level must be non-empty")
        return reference

    @model_validator(mode="after")
    def _coding_has_one_exact_shape(self) -> "ModelTermSpec":
        levels = list(self.levels or ())
        if self.coding == "continuous":
            if self.transform != "identity":
                raise ValueError("continuous model terms require transform='identity'")
            if levels or self.reference_level is not None:
                raise ValueError(
                    "continuous model terms cannot declare levels or a reference"
                )
            return self

        if self.coding == "ordinal_linear":
            if self.transform != "declared_level_index":
                raise ValueError(
                    "ordinal_linear model terms require "
                    "transform='declared_level_index'"
                )
            if len(levels) < 2:
                raise ValueError(
                    "ordinal_linear model terms require at least two ordered levels"
                )
            if self.reference_level is not None:
                raise ValueError(
                    "ordinal_linear model terms do not use a reference level"
                )
            return self

        if self.transform != "treatment_contrast":
            raise ValueError(
                f"{self.coding} model terms require transform='treatment_contrast'"
            )
        minimum = 2
        if len(levels) < minimum:
            raise ValueError(
                f"{self.coding} model terms require at least {minimum} levels"
            )
        if self.coding == "binary" and len(levels) != 2:
            raise ValueError("binary model terms require exactly two levels")
        if self.reference_level not in levels:
            raise ValueError(
                "treatment-coded model term reference_level must be one of levels"
            )
        return self

    @property
    def contrast_levels(self) -> tuple[str, ...]:
        """Non-reference levels in the Planner-declared order."""

        return tuple(
            level for level in (self.levels or ()) if level != self.reference_level
        )


def validate_model_term_roster(
    *,
    terms: Optional[Sequence[ModelTermSpec]],
    exposure: str,
    covariates: Optional[Sequence[str]],
) -> tuple[ModelTermSpec, tuple[ModelTermSpec, ...]]:
    """Validate one exposure plus the exact ordered adjustment roster.

    The helper intentionally requires ``terms`` rather than synthesising them
    from names. A names-only legacy plan remains readable, but cannot authorize
    a deterministic scientific fit whose coding was never declared.
    """

    if terms is None:
        raise ValueError("model_terms must explicitly declare variable coding")
    roster = tuple(terms)
    names = [item.name for item in roster]
    if len(names) != len(set(names)):
        raise ValueError("model_terms must not repeat a source variable")
    exposures = [item for item in roster if item.role == "exposure"]
    if len(exposures) != 1 or exposures[0].name != exposure:
        raise ValueError(
            "model_terms must contain exactly one exposure matching exposure_source"
        )
    adjustments = tuple(item for item in roster if item.role == "covariate")
    declared_covariates = tuple(item.name for item in adjustments)
    if covariates is not None and declared_covariates != tuple(covariates):
        raise ValueError(
            "model_terms covariate order must exactly match the declared covariates"
        )
    return exposures[0], adjustments


def serialise_model_terms(terms: Iterable[ModelTermSpec]) -> list[dict[str, object]]:
    """Stable JSON-ready representation shared by scaffolds and receipts."""

    return [term.model_dump(mode="json") for term in terms]


__all__ = [
    "ModelTermCoding",
    "ModelTermRole",
    "ModelTermSpec",
    "ModelTermTransform",
    "level_spelling",
    "serialise_model_terms",
    "validate_model_term_roster",
]
