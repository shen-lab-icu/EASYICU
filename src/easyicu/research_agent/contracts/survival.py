"""Host-owned contracts for a reportable primary survival execution.

The Planner chooses the estimand in ``FamilyPrimaryResultRequirement``.  This
module owns the receipt that can only be issued by EasyICU's sealed survival
executor after it has read a digest-bound cohort, fitted the declared Cox
model, materialised the result and executed the declared PH diagnostic.
"""

from __future__ import annotations

import re
from typing import Dict, List, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .model_terms import ModelTermSpec, validate_model_term_roster
from .model_tokens import (
    SURVIVAL_COX_ESTIMATOR,
    SURVIVAL_PH_DIAGNOSTIC,
)


SURVIVAL_ANALYSIS_RECEIPT_PRODUCT = "log:survival_analysis_receipt"
SURVIVAL_PH_DIAGNOSTIC_PRODUCT = "table:survival_ph_diagnostic"
SURVIVAL_PRIMARY_OWNER = "easyicu.host.survival_primary_cox_v1"


def canonical_survival_formula(
    *,
    time_column: str,
    event_column: str,
    event_value: int,
    exposure_source: str,
    covariates: Sequence[str],
    design_columns: Sequence[str] = (),
) -> str:
    """Render the one formula representation owned by the Cox executor."""

    return f"Surv({time_column}, {event_column}=={int(event_value)}) ~ " + " + ".join(
        list(design_columns) or [exposure_source, *covariates]
    )


def canonical_survival_applied_filter(
    *,
    time_column: str,
    event_column: str,
    event_value: int,
    exposure_source: str,
    covariates: Sequence[str],
    model_terms: Sequence[ModelTermSpec],
    time_horizon_value: float,
    time_unit: str,
) -> str:
    """Describe the exact host transformation without copying policy at gates."""

    needed = [time_column, event_column, exposure_source, *covariates]
    term_summary = ",".join(
        f"{term.name}:{term.coding}:{term.transform}" for term in model_terms
    )
    return (
        "strict_numeric(" + ",".join([time_column, event_column]) + "); "
        f"declared_model_terms({term_summary}); "
        "complete_case(" + ",".join(needed) + "); time>0; "
        f"{event_column}=({event_column}=={int(event_value)}); "
        f"administrative_censor_at={float(time_horizon_value):g}_{time_unit}"
    )


class SurvivalAnalysisReceipt(BaseModel):
    """Digest-bound receipt issued by the deterministic survival owner."""

    model_config = ConfigDict(extra="forbid")

    issuer: Literal["easyicu.host.survival_primary_cox_v1"] = SURVIVAL_PRIMARY_OWNER
    execution_mode: Literal["deterministic_standard"] = "deterministic_standard"
    result_product: str
    result_evidence_id: str
    result_sha256: str
    input_product: str
    input_evidence_id: str
    input_sha256: str
    analysis_frame_sha256: str
    ph_diagnostic_product: Literal["table:survival_ph_diagnostic"] = (
        SURVIVAL_PH_DIAGNOSTIC_PRODUCT
    )
    ph_diagnostic_evidence_id: str
    ph_diagnostic_sha256: str
    exposure_source: str
    outcome: str
    effect_scale: str
    analysis_population: str
    n_source_rows: int = Field(ge=1)
    n_analysis_rows: int = Field(ge=1)
    n_complete_case_dropped: int = Field(ge=0)
    n_censored_at_horizon: int = Field(ge=0)
    n_events: int = Field(ge=1)
    time_origin: str
    time_column: str
    time_unit: Literal["minutes", "hours", "days"]
    event_column: str
    event_value: int
    censor_value: Literal[0] = 0
    event_definition: str
    censoring_strategy: str
    competing_risk_strategy: str
    time_horizon: str
    time_horizon_value: float = Field(gt=0)
    estimator: str
    effect_measure: str
    formula: str
    covariates: List[str]
    model_terms: List[ModelTermSpec]
    design_columns: List[str]
    exposure_design_column: str
    applied_filter: str
    time_unit_authority: Literal["research_context_concept_descriptor"] = (
        "research_context_concept_descriptor"
    )
    package_versions: Dict[str, str]
    proportional_hazards_diagnostic: str
    proportional_hazards_tested: Literal[True] = True
    proportional_hazards_p_value: float = Field(ge=0, le=1)
    proportional_hazards_alpha: float = Field(gt=0, lt=1)
    proportional_hazards_policy: Literal[
        "report_only", "block_paper_authorization", "human_review"
    ]
    proportional_hazards_status: Literal[
        "not_rejected",
        "violation_report_only",
        "violation_block_paper_authorization",
        "violation_human_review",
    ]
    paper_authorization_allowed: bool

    @field_validator(
        "result_product",
        "result_evidence_id",
        "input_product",
        "input_evidence_id",
        "exposure_source",
        "outcome",
        "effect_scale",
        "analysis_population",
        "time_origin",
        "time_column",
        "event_column",
        "event_definition",
        "censoring_strategy",
        "competing_risk_strategy",
        "time_horizon",
        "estimator",
        "effect_measure",
        "formula",
        "exposure_design_column",
        "applied_filter",
        "proportional_hazards_diagnostic",
        "ph_diagnostic_evidence_id",
    )
    @classmethod
    def _require_nonblank_text(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("survival analysis receipt text fields must be non-empty")
        return text

    @field_validator(
        "result_sha256",
        "input_sha256",
        "analysis_frame_sha256",
        "ph_diagnostic_sha256",
    )
    @classmethod
    def _require_sha256(cls, value: str) -> str:
        digest = str(value or "").lower()
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("survival analysis receipt digests must be SHA-256")
        return digest

    @field_validator("covariates")
    @classmethod
    def _require_unique_covariates(cls, value: List[str]) -> List[str]:
        names = [str(item or "").strip() for item in value]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("survival covariates must be unique and nonblank")
        return names

    @field_validator("design_columns")
    @classmethod
    def _require_unique_design_columns(cls, value: List[str]) -> List[str]:
        names = [str(item or "").strip() for item in value]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("survival design columns must be unique and nonblank")
        return names

    @field_validator("package_versions")
    @classmethod
    def _require_package_versions(cls, value: Dict[str, str]) -> Dict[str, str]:
        required = {"easyicu", "lifelines", "pandas"}
        if not required.issubset(value) or any(
            not str(name).strip() or not str(version).strip()
            for name, version in value.items()
        ):
            raise ValueError(
                "survival receipt must bind nonblank easyicu/lifelines/pandas versions"
            )
        return dict(value)

    @model_validator(mode="after")
    def _validate_execution_relationships(self) -> "SurvivalAnalysisReceipt":
        if self.n_analysis_rows > self.n_source_rows:
            raise ValueError("n_analysis_rows cannot exceed n_source_rows")
        if self.n_events > self.n_analysis_rows:
            raise ValueError("n_events cannot exceed n_analysis_rows")
        if self.n_complete_case_dropped != self.n_source_rows - self.n_analysis_rows:
            raise ValueError(
                "n_complete_case_dropped must reconcile source and analysis rows"
            )
        if not self.result_product.startswith("table:"):
            raise ValueError("result_product must name the materialised result table")
        if self.result_evidence_id != f"sha256:{self.result_sha256}":
            raise ValueError(
                "result_evidence_id must be content-addressed by result SHA"
            )
        if self.ph_diagnostic_evidence_id != f"sha256:{self.ph_diagnostic_sha256}":
            raise ValueError(
                "ph_diagnostic_evidence_id must be content-addressed by diagnostic SHA"
            )
        expected_formula = canonical_survival_formula(
            time_column=self.time_column,
            event_column=self.event_column,
            event_value=self.event_value,
            exposure_source=self.exposure_source,
            covariates=self.covariates,
            design_columns=self.design_columns,
        )
        if self.formula != expected_formula:
            raise ValueError(
                "survival receipt formula is not the canonical host formula"
            )
        expected_filter = canonical_survival_applied_filter(
            time_column=self.time_column,
            event_column=self.event_column,
            event_value=self.event_value,
            exposure_source=self.exposure_source,
            covariates=self.covariates,
            model_terms=self.model_terms,
            time_horizon_value=self.time_horizon_value,
            time_unit=self.time_unit,
        )
        if self.applied_filter != expected_filter:
            raise ValueError("survival receipt filter is not the canonical host filter")
        if self.estimator != SURVIVAL_COX_ESTIMATOR:
            raise ValueError("the host survival receipt currently supports Cox only")
        if self.proportional_hazards_diagnostic != SURVIVAL_PH_DIAGNOSTIC:
            raise ValueError(
                "the host survival receipt requires the exact global Schoenfeld test"
            )
        exposure_term, _ = validate_model_term_roster(
            terms=self.model_terms,
            exposure=self.exposure_source,
            covariates=self.covariates,
        )
        if exposure_term.coding == "categorical":
            raise ValueError(
                "the v1 survival primary result has no multi-contrast exposure shape"
            )
        if self.exposure_design_column not in self.design_columns:
            raise ValueError("exposure_design_column must be in design_columns")
        violation = self.proportional_hazards_p_value < self.proportional_hazards_alpha
        if not violation:
            expected_status = "not_rejected"
        elif self.proportional_hazards_policy == "report_only":
            expected_status = "violation_report_only"
        elif self.proportional_hazards_policy == "block_paper_authorization":
            expected_status = "violation_block_paper_authorization"
        else:
            expected_status = "violation_human_review"
        if self.proportional_hazards_status != expected_status:
            raise ValueError("PH receipt status does not follow the declared policy")
        expected_authorization = expected_status in {
            "not_rejected",
            "violation_report_only",
        }
        if self.paper_authorization_allowed != expected_authorization:
            raise ValueError(
                "PH paper_authorization_allowed does not follow the declared policy"
            )
        if self.exposure_source in self.covariates:
            raise ValueError("covariates must not repeat the exposure")
        if self.time_column in self.covariates or self.event_column in self.covariates:
            raise ValueError("covariates must not contain time/event columns")
        return self


__all__ = [
    "SURVIVAL_ANALYSIS_RECEIPT_PRODUCT",
    "SURVIVAL_PH_DIAGNOSTIC_PRODUCT",
    "SURVIVAL_PRIMARY_OWNER",
    "SurvivalAnalysisReceipt",
    "canonical_survival_applied_filter",
    "canonical_survival_formula",
]
