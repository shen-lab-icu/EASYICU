"""Typed, host-derived authority for qualitative scientific claims.

Evidence citations prove where an artefact came from.  They do not prove that
arbitrary prose accurately describes that artefact.  This module owns the
small public contract that closes that gap for qualitative association claims:
a reviewed deterministic executor emits machine-readable model fields, the
host derives a claim from those fields, and the Writer may select only the
resulting claim reference.  Neither an execution script nor the Writer can
author the scientific sentence.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class ScientificClaimDraft(BaseModel):
    """One machine-readable association claim derived by the host."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["easyicu.scientific_claim/1"] = (
        "easyicu.scientific_claim/1"
    )
    claim_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    claim_type: Literal["association"]
    exposure: str
    outcome: str
    direction: Literal["positive", "negative", "no_clear_association"]
    estimand: str
    population: str
    analysis_role: Literal["primary", "secondary", "sensitivity", "exploratory"]
    status: Literal["supported"]
    adjusted_for: list[str] = Field(default_factory=list)

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

        if self.direction == "positive":
            relation = "was positively associated with"
        elif self.direction == "negative":
            relation = "was negatively associated with"
        else:
            relation = "showed no clear association with"
        adjustment = ""
        if self.adjusted_for:
            adjustment = "After adjustment for " + ", ".join(self.adjusted_for) + ", "
        return (
            f"{adjustment}{self.exposure} {relation} {self.outcome} in "
            f"{self.population} (estimand: {self.estimand}; analysis role: "
            f"{self.analysis_role})."
        )


def scientific_claim_compilation_requested(summary: object) -> bool:
    """Return whether a summary belongs to a supported host compiler."""

    if not isinstance(summary, dict):
        return False
    if "scientific_claims" in summary:
        raise ValueError(
            "scientific_claims are host-derived and must not be supplied by a runner"
        )
    return str(summary.get("interpretation_class") or "").strip() == (
        "adjusted_association"
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

    def _required_text(field: str) -> str:
        value = " ".join(str(summary.get(field) or "").split())
        if not value:
            raise ValueError(
                f"scientific_claims cannot be derived without {field!r}"
            )
        return value

    effect_scale = _required_text("effect_scale").lower()
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
    if low > null_value:
        direction = "positive"
    elif high < null_value:
        direction = "negative"
    else:
        direction = "no_clear_association"

    role = _required_text("analysis_role").lower()
    if role not in {"primary", "secondary", "sensitivity", "exploratory"}:
        raise ValueError(
            "scientific_claims adjusted-association analysis_role is unsupported"
        )
    analysis_set = _required_text("analysis_set").replace("_", " ")
    covariates = summary.get("adjustment_covariates", summary.get("covariates", []))
    if not isinstance(covariates, list):
        raise ValueError("scientific_claims adjustment covariates must be a list")

    return [
        ScientificClaimDraft(
            claim_id="adjusted_association",
            claim_type="association",
            exposure=_required_text("exposure"),
            outcome=_required_text("outcome"),
            direction=direction,
            estimand=estimand,
            population=f"the {analysis_set} analysis set",
            analysis_role=role,
            status="supported",
            adjusted_for=[str(value) for value in covariates],
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
