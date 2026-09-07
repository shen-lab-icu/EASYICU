"""Typed, dependency-neutral scientific model-term declarations.

The Planner owns how each exposure and adjustment variable enters a model.
Executors consume this contract; they must not infer categorical treatment,
level ordering, or reference levels from pandas dtypes or column names.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .dependence import PlannedDependenceRequirement
from .model_tokens import (
    ASSOCIATION_GLM_BINOMIAL_ESTIMATOR,
    ASSOCIATION_LOGIT_ESTIMATOR,
    ASSOCIATION_OLS_ESTIMATOR,
    canonical_association_method as _canonical_association_method,
    normalise_model_contract_token as _normalise_model_contract_token,
)


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


def level_identity_class(value: object) -> str:
    """The type class a level spelling stands for.

    ``level_spelling`` is deliberately lossy -- it is the *wire* identity, and
    collapsing ``1`` and ``1.0`` is correct because they are the same value.
    Collapsing ``1`` and ``"1"``, or ``True`` and ``"true"``, is not: those are
    two source categories, and treatment coding them as one merges two groups
    into a single contrast with nothing recorded.

    That is not hypothetical for this codebase. ``io.data_converter`` pins
    ``MIXED_TYPE_COLUMNS`` to string precisely because real EHR exports carry
    columns whose object dtype holds both, and a cohort assembled from more
    than one source can reach an executor with the mixture intact.

    Returning a class rather than a richer typed level keeps the declared
    contract, every receipt and every digest byte-identical: the ambiguity is
    detected where declaration meets data, instead of by rewriting the wire
    format of every model term.
    """

    if value is None:
        return "missing"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, float) and value != value:  # NaN
        return "missing"
    if isinstance(value, (int, float)):
        return "numeric"
    if isinstance(value, str):
        return "text"
    return type(value).__name__


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


# The v1 planner-owned roster is intentionally narrower than EasyICU's full
# modeling capability.  These are the adjusted-association families whose
# execution contract is currently checked by PrimaryModelContractValidator.
# Survival, prediction, mixed-effects, and clustering methods keep their own
# family-specific plans/contracts until an equally typed validator exists.
ADJUSTED_ASSOCIATION_BINARY_METHOD_FAMILIES = frozenset(
    {
        ASSOCIATION_LOGIT_ESTIMATOR,
        ASSOCIATION_GLM_BINOMIAL_ESTIMATOR,
    }
)
ADJUSTED_ASSOCIATION_CONTINUOUS_METHOD_FAMILIES = frozenset(
    {
        ASSOCIATION_OLS_ESTIMATOR,
        "statsmodels_quantreg",
        "statsmodels_quantreg_median_vcov_robust",
    }
)


class PlannedModelRequirement(BaseModel):
    """Planner-owned obligation for a supported adjusted-association model.

    The planner chooses these scientific commitments.  Execution validators
    only reconcile the emitted model contracts against this typed roster; they
    must not infer required models from step prose or benchmark vocabulary.
    This v1 schema does not represent survival, prediction, mixed-effects, or
    clustering contracts.
    """

    model_config = ConfigDict(extra="forbid")

    requirement_id: str
    outcome: str
    outcome_type: Literal["binary", "continuous"]
    method_family: str
    exposure_source: str
    analysis_role: Literal["primary", "secondary", "sensitivity"]
    analysis_set: Literal["source_aware", "complete_case"]
    required_for_step_success: bool = True
    covariates: Optional[List[str]] = Field(
        default=None,
        description=(
            "The exact adjustment set, or null when the planner did not declare "
            "one. An empty list is a declaration of an unadjusted model, which "
            "is not the same statement as null."
        ),
    )
    covariate_rationales: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Planner-owned confounding rationale for each selected covariate. "
            "For an exact user-owned roster this is copied from sealed study "
            "authority; for a Planner-selectable roster it is proposed and "
            "reviewed as part of the complete plan."
        ),
    )
    covariate_temporal_roles: Dict[
        str, Literal["baseline_static", "at_or_before_time_zero"]
    ] = Field(
        default_factory=dict,
        description=(
            "Typed pre-time-zero eligibility for every selected covariate. "
            "Availability alone is not temporal adjustment authority."
        ),
    )
    model_terms: Optional[List[ModelTermSpec]] = Field(
        default=None,
        description=(
            "The exact exposure and covariate coding contract. Legacy plans "
            "without this field remain readable but cannot authorize a "
            "deterministic host fit."
        ),
    )
    exposure_levels: Optional[List[str]] = Field(
        default=None,
        description=(
            "The closed, ordered level set of a categorical or ordinal exposure, "
            "or null for a binary or continuous one. Declaring it commits the "
            "model to one contrast per non-reference level."
        ),
    )
    exposure_reference_level: Optional[str] = Field(
        default=None,
        description="Which declared level every contrast is taken against.",
    )
    primary_contrast_level: Optional[str] = Field(
        default=None,
        description=(
            "Which contrast is the headline estimate the manuscript reports. "
            "With more than two levels this cannot be inferred: the highest "
            "level against the reference and a per-level trend are different "
            "scientific claims, and choosing between them is the planner's."
        ),
    )
    dependence: Optional[PlannedDependenceRequirement] = Field(
        default=None,
        description=(
            "Exact repeated-unit covariance contract bound from StudyContext "
            "authority. Null means model-based covariance; execution must not "
            "infer clustering from intent text or identifier-like column names."
        ),
    )

    @field_validator(
        "requirement_id",
        "outcome",
        "exposure_source",
    )
    @classmethod
    def _nonblank_text(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("planned model requirement fields must be non-empty")
        return text

    @field_validator("method_family", mode="before")
    @classmethod
    def _canonical_method_family(cls, value: object) -> str:
        method = _canonical_association_method(value)
        if not method:
            raise ValueError("planned model requirement fields must be non-empty")
        return method

    @field_validator("covariates")
    @classmethod
    def _exact_unique_covariate_names(
        cls, value: Optional[List[str]]
    ) -> Optional[List[str]]:
        """An adjustment set is a roster of exact columns or it is not declared.

        ``None`` stays ``None``: "the planner did not say" must remain
        distinguishable from "the planner said none", because a host that reads
        the first as the second would fit an unadjusted model and label it the
        pre-specified adjusted one.
        """

        if value is None:
            return None
        names = [str(item or "").strip() for item in value]
        if any(not name for name in names):
            raise ValueError("covariates must not contain blank names")
        if len(names) != len(set(names)):
            raise ValueError("covariates must not repeat a name")
        return names

    @field_validator("covariate_rationales")
    @classmethod
    def _bounded_planned_covariate_rationales(
        cls, value: Dict[str, str]
    ) -> Dict[str, str]:
        cleaned: Dict[str, str] = {}
        for raw_name, raw_rationale in value.items():
            name = str(raw_name or "").strip()
            rationale = str(raw_rationale or "").strip()
            if not name or len(rationale) < 8 or len(rationale) > 600:
                raise ValueError(
                    "planned covariate rationales require non-empty names and "
                    "8-600 character explanations"
                )
            cleaned[name] = rationale
        return cleaned

    @model_validator(mode="after")
    def _method_family_matches_supported_outcome(self) -> "PlannedModelRequirement":
        supported = (
            ADJUSTED_ASSOCIATION_BINARY_METHOD_FAMILIES
            if self.outcome_type == "binary"
            else ADJUSTED_ASSOCIATION_CONTINUOUS_METHOD_FAMILIES
        )
        method_family = _normalise_model_contract_token(self.method_family)
        if method_family not in supported:
            raise ValueError(
                "model_requirements currently support only binary logistic "
                "or continuous linear/quantile adjusted-association families; "
                f"outcome_type={self.outcome_type!r} is incompatible with "
                f"method_family={self.method_family!r}"
            )
        if (
            self.analysis_role in {"primary", "secondary"}
            and not self.required_for_step_success
        ):
            raise ValueError(
                "primary and secondary model_requirements must be required "
                "for step success; only a sensitivity requirement may be optional"
            )
        if self.covariates is not None:
            # Two adjustment sets that are wrong on their face, and wrong in a
            # way the contract can see without knowing the case.  Conditioning
            # on the outcome, or on the exposure whose effect is being
            # estimated, does not produce a weaker version of the declared
            # estimand -- it produces a different quantity that would still be
            # reported under the declared one's name.
            if self.outcome in self.covariates:
                raise ValueError(
                    "covariates must not contain the outcome "
                    f"{self.outcome!r}; conditioning on the outcome does not "
                    "estimate the declared association"
                )
            if self.exposure_source in self.covariates:
                raise ValueError(
                    "covariates must not contain the exposure "
                    f"{self.exposure_source!r}; adjusting for the exposure "
                    "removes the association the requirement declares"
                )
        rationale_keys = set(self.covariate_rationales)
        temporal_keys = set(self.covariate_temporal_roles)
        if rationale_keys or temporal_keys:
            expected_covariates = set(self.covariates or ())
            if rationale_keys != expected_covariates or temporal_keys != expected_covariates:
                raise ValueError(
                    "planned covariate rationale and temporal-role maps must "
                    "cover exactly the declared covariates"
                )
        if self.model_terms is not None:
            exposure_term, adjustment_terms = validate_model_term_roster(
                terms=self.model_terms,
                exposure=self.exposure_source,
                covariates=self.covariates,
            )
            term_covariates = [item.name for item in adjustment_terms]
            if self.covariates is None:
                self.covariates = term_covariates
            if exposure_term.transform == "treatment_contrast":
                levels = list(exposure_term.levels or ())
                reference = exposure_term.reference_level
                if (
                    self.exposure_levels is not None
                    and list(self.exposure_levels) != levels
                ):
                    raise ValueError(
                        "exposure_levels must match the exposure ModelTermSpec"
                    )
                if (
                    self.exposure_reference_level is not None
                    and self.exposure_reference_level != reference
                ):
                    raise ValueError(
                        "exposure_reference_level must match the exposure ModelTermSpec"
                    )
                self.exposure_levels = levels
                self.exposure_reference_level = reference
                if exposure_term.coding == "binary":
                    only_contrast = exposure_term.contrast_levels[0]
                    if (
                        self.primary_contrast_level is not None
                        and self.primary_contrast_level != only_contrast
                    ):
                        raise ValueError(
                            "a binary exposure's primary contrast is its one "
                            "non-reference level"
                        )
                    self.primary_contrast_level = only_contrast
                elif self.primary_contrast_level is None:
                    raise ValueError(
                        "a categorical exposure ModelTermSpec requires "
                        "primary_contrast_level on the model requirement"
                    )
            elif any(
                value is not None
                for value in (
                    self.exposure_levels,
                    self.exposure_reference_level,
                    self.primary_contrast_level,
                )
            ):
                raise ValueError(
                    "identity/ordinal-linear exposure coding cannot also declare "
                    "treatment-contrast fields"
                )
        self._check_declared_exposure_levels()
        return self

    def _check_declared_exposure_levels(self) -> None:
        """Refuse a partial categorical-exposure contrast declaration."""

        declared = {
            "exposure_levels": self.exposure_levels,
            "exposure_reference_level": self.exposure_reference_level,
            "primary_contrast_level": self.primary_contrast_level,
        }
        present = {name for name, value in declared.items() if value is not None}
        if not present:
            return
        missing = sorted(set(declared) - present)
        if missing:
            raise ValueError(
                "a categorical exposure is declared by "
                + ", ".join(sorted(declared))
                + " together; this requirement is missing "
                + ", ".join(repr(name) for name in missing)
                + ", so the host cannot tell which contrast the manuscript reports"
            )
        levels = [str(value or "").strip() for value in self.exposure_levels or []]
        if any(not level for level in levels):
            raise ValueError("exposure_levels must not contain a blank level")
        if len(levels) != len(set(levels)):
            raise ValueError("exposure_levels must not repeat a level")
        if len(levels) < 2:
            raise ValueError("exposure_levels needs at least two levels")
        reference = str(self.exposure_reference_level or "").strip()
        primary = str(self.primary_contrast_level or "").strip()
        if reference not in levels:
            raise ValueError(
                f"exposure_reference_level {reference!r} is not one of the "
                "declared exposure_levels"
            )
        if primary not in levels:
            raise ValueError(
                f"primary_contrast_level {primary!r} is not one of the "
                "declared exposure_levels"
            )
        if primary == reference:
            raise ValueError("primary_contrast_level must not be the reference level")


__all__ = [
    "ADJUSTED_ASSOCIATION_BINARY_METHOD_FAMILIES",
    "ADJUSTED_ASSOCIATION_CONTINUOUS_METHOD_FAMILIES",
    "PlannedModelRequirement",
    "ModelTermCoding",
    "ModelTermRole",
    "ModelTermSpec",
    "ModelTermTransform",
    "level_identity_class",
    "level_spelling",
    "serialise_model_terms",
    "validate_model_term_roster",
]
