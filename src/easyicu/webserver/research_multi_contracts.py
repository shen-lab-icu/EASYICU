"""Bounded host/browser contracts; none enter the Planner output schema."""
from __future__ import annotations

from typing import Annotated, Literal
import re

from pydantic import BaseModel, ConfigDict, Field, model_validator

from easyicu.databases.research_requirements import CapabilityDecision, ThresholdRequirement
from easyicu.research_agent.canonical_json import canonical_sha256

Code = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")]
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Database = Literal["miiv", "mimic", "eicu", "hirid", "aumc", "sic", "miiv_demo", "mimic_demo", "eicu_demo"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatabaseTarget(Contract):
    database: Database
    source_id: Code | None = None


class MultiDatabaseRequest(Contract):
    project_id: Code
    session_id: Code
    study_context_id: Code
    source_run_id: Code | None = None
    targets: tuple[DatabaseTarget, ...] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def unique_databases(self):
        if len({row.database for row in self.targets}) != len(self.targets):
            raise ValueError("multi_database_duplicate_target")
        return self


class EgressGrant(Contract):
    source_id: Code
    source_sha256: Digest
    database: Database
    provider_sha256: Digest
    data_handling: Literal["on_device", "zero_retention_declared", "user_accepted_external", "unspecified"]
    clearance: Literal["provider_required", "verified_public_or_synthetic"]
    grants: tuple[Literal["prepare_source", "review_plan", "execute_plan", "provider_egress"], ...] = Field(min_length=1, max_length=4)


class ReviewedRun(Contract):
    run_id: Code
    study_context_id: Code
    plan_sha256: Digest
    context_sha256: Digest
    scientific_review_sha256: Digest
    requests_sha256: Digest
    input_sha256: Digest
    configuration_sha256: Digest


class SubstitutionView(Contract):
    capability: Code
    reason_code: Code
    original: Code | ThresholdRequirement
    substitute: Code | ThresholdRequirement


class TargetCard(Contract):
    database: Database
    source_id: Code | None = None
    source_sha256: Digest | None = None
    disposition: Literal["exact", "adjusted", "not_run"]
    decision_sha256: Digest
    substitutions: tuple[SubstitutionView, ...] = Field(default=(), max_length=64)
    definition_codes: tuple[Code, ...] = Field(default=(), max_length=64)
    reason_codes: tuple[Code, ...] = Field(default=(), max_length=64)
    review: ReviewedRun | None = None
    egress: EgressGrant | None = None


class PreparedTarget(Contract):
    """Host-only binding. A browser may never supply this object."""
    target: DatabaseTarget
    decision: CapabilityDecision
    source_sha256: Digest | None = None
    review: ReviewedRun | None = None
    egress: EgressGrant | None = None
    stop_code: Code | None = None
    definition_codes: tuple[Code, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def bound(self):
        if self.decision.database != self.target.database:
            raise ValueError("multi_database_binding_changed")
        if self.review is not None and (self.decision.status == "refusal" or self.stop_code or not self.egress):
            raise ValueError("multi_database_invalid_review")
        if self.egress and (self.egress.database != self.target.database
                            or self.egress.source_id != self.target.source_id
                            or self.egress.source_sha256 != self.source_sha256):
            raise ValueError("multi_database_egress_changed")
        return self

    def card(self):
        return TargetCard(database=self.target.database, source_id=self.target.source_id,
            source_sha256=self.source_sha256, decision_sha256=canonical_sha256(self.decision.model_dump(mode="json")),
            disposition=("adjusted" if self.decision.substitutions or self.definition_codes else "exact") if self.review else "not_run",
            definition_codes=self.definition_codes,
            substitutions=tuple(SubstitutionView(**{key: getattr(row, key) for key in
                ("capability", "reason_code", "original", "substitute")}) for row in self.decision.substitutions),
            reason_codes=tuple(dict.fromkeys([*(row.code for row in self.decision.missing),
                *([self.stop_code] if self.stop_code else [])])), review=self.review, egress=self.egress)


class CombinedCard(Contract):
    schema_version: Literal["easyicu.multi_database_card/1"] = "easyicu.multi_database_card/1"
    batch_id: Digest
    request_sha256: Digest
    design_sha256: Digest
    execution: Literal["sequential"] = "sequential"
    result_scope: Literal["analysis_level"] = "analysis_level"
    targets: tuple[TargetCard, ...] = Field(min_length=1, max_length=6)

    @property
    def digest(self):
        return canonical_sha256(self.model_dump(mode="json"))


class MultiDatabaseError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def failure_code(error, default="multi_database_execution_failed"):
    for key in ("code", "issue", "reason_code"):
        value = getattr(error, key, None)
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}", value):
            return value
    detail = getattr(error, "detail", {})
    value = detail.get("error") if isinstance(detail, dict) else None
    return value if isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,159}", value) else default
