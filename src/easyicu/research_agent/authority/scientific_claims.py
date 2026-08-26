"""Typed, host-derived authority for qualitative scientific claims.

Evidence citations prove where an artefact came from.  They do not prove that
arbitrary prose accurately describes that artefact.  This module owns the
small public contract that closes that gap for reviewed association and
descriptive claims: a deterministic executor emits machine-readable fields,
the host derives a claim from those fields, and the Writer may select only the
resulting claim reference.  Neither an execution script nor the Writer can
author the scientific sentence.
"""

from __future__ import annotations

import math
import re
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


class ScientificClaimDraft(BaseModel):
    """One machine-readable scientific claim derived by the host."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        "easyicu.scientific_claim/1", "easyicu.scientific_claim/2"
    ] = "easyicu.scientific_claim/1"
    claim_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    claim_type: Literal[
        "association",
        "descriptive_absolute_risk",
        "descriptive_risk_difference",
    ]
    exposure: str
    outcome: str
    direction: Literal[
        "positive",
        "negative",
        "no_clear_association",
        "descriptive_only",
    ]
    estimand: str
    population: str
    analysis_role: Literal[
        "primary", "secondary", "sensitivity", "exploratory", "auxiliary"
    ]
    status: Literal["supported"]
    adjusted_for: list[str] = Field(default_factory=list)
    point_estimate: float | None = None
    interval_lower: float | None = None
    interval_upper: float | None = None

    @field_validator("exposure", "outcome", "estimand", "population")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = " ".join(str(value or "").split())
        if not normalized:
            raise ValueError("scientific claim fields must be non-empty")
        return normalized

    @field_validator("adjusted_for")
    @classmethod
    def _unique_adjustment_terms(cls, values: list[str]) -> list[str]:
        normalized = [" ".join(str(value or "").split()) for value in values]
        if any(not value for value in normalized) or len(set(normalized)) != len(
            normalized
        ):
            raise ValueError(
                "scientific claim adjusted_for must contain unique non-empty terms"
            )
        return normalized

    @model_validator(mode="after")
    def _claim_kind_matches_its_ceiling(self) -> "ScientificClaimDraft":
        numeric_values = (
            self.point_estimate,
            self.interval_lower,
            self.interval_upper,
        )
        if any(value is not None for value in numeric_values):
            if any(value is None for value in numeric_values):
                raise ValueError(
                    "scientific claim numeric estimate and interval must be complete"
                )
            point, lower, upper = (float(value) for value in numeric_values)
            if not all(math.isfinite(value) for value in (point, lower, upper)):
                raise ValueError("scientific claim numeric values must be finite")
            if lower > point or point > upper:
                raise ValueError(
                    "scientific claim interval must contain its point estimate"
                )
        if self.claim_type == "association":
            if self.schema_version != "easyicu.scientific_claim/1":
                raise ValueError("association claims require scientific_claim/1")
            if self.direction == "descriptive_only":
                raise ValueError("association claims require an association direction")
            if self.analysis_role == "auxiliary":
                raise ValueError("association claims cannot use the auxiliary role")
            return self
        if self.schema_version != "easyicu.scientific_claim/2":
            raise ValueError("descriptive claims require scientific_claim/2")
        if self.direction != "descriptive_only" or self.adjusted_for:
            raise ValueError(
                "descriptive claims must be descriptive_only and cannot claim adjustment"
            )
        return self


class ScientificClaim(ScientificClaimDraft):
    """A validated draft bound by the host to one step and evidence record."""

    step_id: str
    evidence_id: str

    @field_validator("step_id", "evidence_id")
    @classmethod
    def _non_empty_authority_coordinate(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("scientific claim authority coordinates must be non-empty")
        return normalized

    @property
    def claim_ref(self) -> str:
        return f"{self.step_id}.{self.claim_id}"

    @property
    def placeholder(self) -> str:
        return "{claim:" + self.claim_ref + "}"

    def render_text(self) -> str:
        """Render the only manuscript sentence authorized by this contract."""

        if self.claim_type != "association":
            return (
                f"In {self.population}, the {self.estimand} for {self.outcome} at "
                f"{self.exposure}; this is a descriptive, unadjusted, noncausal "
                f"estimate (analysis role: {self.analysis_role})."
            )
        if self.direction == "positive":
            relation = "was positively associated with"
        elif self.direction == "negative":
            relation = "was negatively associated with"
        else:
            relation = "showed no clear association with"
        adjustment = ""
        if self.adjusted_for:
            adjustment = "After adjustment for " + ", ".join(self.adjusted_for) + ", "
        estimate_text = f"estimand: {self.estimand}"
        if self.point_estimate is not None:
            estimate_text = (
                f"{self.estimand}, {self.point_estimate:.6g}; 95% CI, "
                f"{self.interval_lower:.6g} to {self.interval_upper:.6g}"
            )
        return (
            f"{adjustment}{self.exposure} {relation} {self.outcome} in "
            f"{self.population} ({estimate_text}; analysis role: "
            f"{self.analysis_role})."
        )

    def render_reader_text(self) -> str:
        """Render the same claim as publication-scale reader-facing prose.

        ``render_text`` remains the exact machine-authority representation used
        to validate Writer claim tokens.  This projection deliberately omits
        runtime roles, analysis-set identifiers, and raw variable names while
        retaining the claim type, direction, estimate, interval, and causal
        ceiling.  The immutable claim object and its evidence coordinates are
        unchanged.
        """

        def display_number(value: float) -> str:
            return f"{value:.3f}".rstrip("0").rstrip(".")

        def estimand_interval() -> tuple[float, float, float] | None:
            match = re.search(
                r"\bwas\s+([-+]?\d+(?:\.\d+)?)\s+"
                r"(?:percent|percentage points)\s+\("
                r"[^)]*?\bCI,\s+([-+]?\d+(?:\.\d+)?)\s+to\s+"
                r"([-+]?\d+(?:\.\d+)?)",
                self.estimand,
                flags=re.I,
            )
            if match is None:
                return None
            point, lower, upper = (float(value) for value in match.groups())
            return point, lower, upper

        if self.claim_type == "descriptive_absolute_risk":
            values = (
                (self.point_estimate, self.interval_lower, self.interval_upper)
                if self.point_estimate is not None
                else estimand_interval()
            )
            if values is None:
                return (
                    f"In the prespecified group, the {self.estimand}; this was "
                    "a descriptive, unadjusted, noncausal estimate."
                )
            point, lower, upper = values
            assert lower is not None
            assert upper is not None
            return (
                "The observed absolute risk in the prespecified group was "
                f"{display_number(point)}% (95% CI, "
                f"{display_number(lower)}% to "
                f"{display_number(upper)}%); this was a "
                "descriptive, unadjusted, noncausal estimate."
            )
        if self.claim_type == "descriptive_risk_difference":
            values = (
                (self.point_estimate, self.interval_lower, self.interval_upper)
                if self.point_estimate is not None
                else estimand_interval()
            )
            if values is None:
                return (
                    f"The {self.estimand}; this was a descriptive, unadjusted, "
                    "noncausal contrast."
                )
            point, lower, upper = values
            assert lower is not None
            assert upper is not None
            return (
                "The prespecified unadjusted risk difference between groups "
                f"was {display_number(point)} percentage points "
                f"(95% CI, {display_number(lower)} to "
                f"{display_number(upper)}); this was a "
                "descriptive, unadjusted, noncausal contrast."
            )

        if self.direction == "positive":
            relation = "was positively associated with"
        elif self.direction == "negative":
            relation = "was negatively associated with"
        else:
            relation = "showed no clear association with"
        estimate_text = self.estimand
        if self.point_estimate is not None:
            assert self.interval_lower is not None
            assert self.interval_upper is not None
            estimate_text = (
                f"{self.estimand}, {display_number(self.point_estimate)}; "
                f"95% CI, {display_number(self.interval_lower)} to "
                f"{display_number(self.interval_upper)}"
            )
        model_prefix = (
            "In the covariate-adjusted model, " if self.adjusted_for else ""
        )
        return (
            f"{model_prefix}the prespecified exposure {relation} the study "
            f"outcome in the prespecified analysis cohort ({estimate_text})."
        )


def scientific_claim_compilation_requested(summary: object) -> bool:
    """Return whether a summary belongs to a supported host compiler."""

    if not isinstance(summary, dict):
        return False
    if "scientific_claims" in summary:
        raise ValueError(
            "scientific_claims are host-derived and must not be supplied by a runner"
        )
    interpretation_class = str(summary.get("interpretation_class") or "").strip()
    if interpretation_class == "adjusted_association":
        return True
    if interpretation_class != "exposure_outcome_distribution":
        return False
    # Historical auxiliary distribution summaries did not carry qualitative
    # claim authority.  They remain readable.  A new summary that declares the
    # descriptive ceiling or estimate envelope opts into this compiler and is
    # therefore rejected if the rest of that envelope is incomplete.
    return bool(
        "descriptive_estimates" in summary
        or str(summary.get("interpretation_ceiling") or "").strip()
    )


def derive_scientific_claim_drafts(summary: object) -> list[ScientificClaimDraft]:
    """Derive claims from one reviewed deterministic result-summary schema.

    This compiler intentionally recognizes only the host-owned adjusted-
    association executor contract.  An LLM-authored ``scientific_claims`` key
    is rejected instead of becoming self-issued authority.  Unsupported result
    shapes simply expose no qualitative claim to the Writer and therefore fail
    closed if the Writer tries to assert one.
    """

    if not scientific_claim_compilation_requested(summary):
        return []
    assert isinstance(summary, dict)

    if str(summary.get("interpretation_class") or "").strip() == (
        "exposure_outcome_distribution"
    ):
        from .descriptive_scientific_claims import derive_descriptive_claim_payloads

        return [
            ScientificClaimDraft.model_validate(payload)
            for payload in derive_descriptive_claim_payloads(summary)
        ]

    def _required_adjusted_text(field: str) -> str:
        value = " ".join(str(summary.get(field) or "").split())
        if not value:
            raise ValueError(
                f"scientific_claims cannot be derived without {field!r}"
            )
        return value

    effect_scale = _required_adjusted_text("effect_scale").lower()
    if effect_scale == "odds_ratio":
        null_value = 1.0
        estimand = "adjusted odds ratio"
    elif effect_scale == "coefficient":
        null_value = 0.0
        estimand = "adjusted linear-regression coefficient"
    else:
        raise ValueError(
            "scientific_claims adjusted-association effect scale is unsupported"
        )
    interval = summary.get("primary_estimate_interval")
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError(
            "scientific_claims require a two-value primary_estimate_interval"
        )
    try:
        low, high = (float(interval[0]), float(interval[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "scientific_claims primary_estimate_interval must be numeric"
        ) from exc
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(
            "scientific_claims primary_estimate_interval must be finite and ordered"
        )
    raw_point = summary.get("primary_estimate", summary.get("adjusted_effect"))
    point: float | None = None
    if raw_point is not None:
        try:
            point = float(raw_point)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "scientific_claims primary estimate must be numeric"
            ) from exc
        if not math.isfinite(point) or point < low or point > high:
            raise ValueError(
                "scientific_claims primary estimate must be finite and inside its interval"
            )
    if low > null_value:
        direction = "positive"
    elif high < null_value:
        direction = "negative"
    else:
        direction = "no_clear_association"

    role = _required_adjusted_text("analysis_role").lower()
    if role not in {"primary", "secondary", "sensitivity", "exploratory"}:
        raise ValueError(
            "scientific_claims adjusted-association analysis_role is unsupported"
        )
    analysis_set = _required_adjusted_text("analysis_set").replace("_", " ")
    covariates = summary.get("adjustment_covariates", summary.get("covariates", []))
    if not isinstance(covariates, list):
        raise ValueError("scientific_claims adjustment covariates must be a list")

    return [
        ScientificClaimDraft(
            claim_id="adjusted_association",
            claim_type="association",
            exposure=_required_adjusted_text("exposure"),
            outcome=_required_adjusted_text("outcome"),
            direction=direction,
            estimand=estimand,
            population=f"the {analysis_set} analysis set",
            analysis_role=role,
            status="supported",
            adjusted_for=[str(value) for value in covariates],
            point_estimate=point,
            interval_lower=low if point is not None else None,
            interval_upper=high if point is not None else None,
        )
    ]


def bind_scientific_claim_drafts(
    raw_claims: object,
    *,
    step_id: str,
    evidence_id: str,
) -> list[ScientificClaim]:
    """Validate a step-summary claim list and bind host-owned coordinates."""

    if raw_claims is None:
        return []
    if not isinstance(raw_claims, list):
        raise ValueError("scientific_claims must be a list")
    claims: list[ScientificClaim] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_claims):
        if not isinstance(raw, dict):
            raise ValueError(f"scientific_claims[{index}] must be an object")
        try:
            draft = ScientificClaimDraft.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(f"scientific_claims[{index}] is invalid: {exc}") from exc
        if draft.claim_id in seen:
            raise ValueError(
                f"scientific_claims contains duplicate claim_id {draft.claim_id!r}"
            )
        seen.add(draft.claim_id)
        claims.append(
            ScientificClaim(
                **draft.model_dump(),
                step_id=step_id,
                evidence_id=evidence_id,
            )
        )
    return claims


__all__ = [
    "ScientificClaim",
    "ScientificClaimDraft",
    "bind_scientific_claim_drafts",
    "derive_scientific_claim_drafts",
    "scientific_claim_compilation_requested",
]
