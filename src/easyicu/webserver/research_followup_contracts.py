"""Browser contracts for host-owned result suggestions, outside Planner transport."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, ResearchContext
from easyicu.research_agent.planning.family_spec.baseline_association_template import AppendKind
from easyicu.research_agent.authority.baseline_association_runtime import BaselineAssociationRuntimeAuthority

FollowupKind = Literal[
    "prespecified_sensitivity", "functional_form_check", "named_subgroup",
    "dose_response", "complementary_survival", "descriptive_context",
    "complementary_association", "article_extension", "same_study_other_database",
]


@dataclass(frozen=True)
class CompletedResult:
    """Host input loaded from approved, completed, verified result evidence.

    Never construct this from a request body. Context and runtime authority
    stay host-side; the response carries only their content identity.
    """

    run_id: str
    source_sha256: str
    plan: AnalysisPlan
    context: ResearchContext
    products: frozenset[str]
    completed_steps: tuple[AnalysisStep, ...]
    baseline: BaselineAssociationRuntimeAuthority | None = None
    completed_append_kinds: frozenset[str] = frozenset()


class FollowupReading(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_family: str
    database: str
    primary_exposure: str | None
    target_outcome: str | None

    @classmethod
    def from_context(cls, context: ResearchContext, *, family: str):
        return cls(source_context_sha256=canonical_sha256(context.model_dump(mode="json")),
                   analysis_family=family, database=context.cohort.database,
                   primary_exposure=context.primary_exposure, target_outcome=context.target_outcome)


class AppendDelta(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["append"] = "append"
    kind: AppendKind
    plan: AnalysisPlan


class NewStudyDelta(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["new_study"] = "new_study"
    source_reading: FollowupReading
    analysis_family: Literal["association_study", "descriptive_epidemiology", "survival"]
    primary_exposure: str | None
    plan: AnalysisPlan


class DatabaseDecision(BaseModel):
    """A host db6a adapter's decision, bound to the source reading and target."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    database: str
    source_context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["exact", "substitution", "refusal"]
    reason_codes: tuple[str, ...]


class OtherDatabaseDelta(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["other_database"] = "other_database"
    source_reading: FollowupReading
    target_reading: FollowupReading
    capability_decision: DatabaseDecision
    plan: AnalysisPlan


class FollowupSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    suggestion_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    kind: FollowupKind
    target_action: Literal["append", "new_study", "other_database"]
    reason_codes: tuple[str, ...]
    delta: AppendDelta | NewStudyDelta | OtherDatabaseDelta = Field(discriminator="action")


class ResultFollowups(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.result_followups/1"] = "easyicu.result_followups/1"
    run_id: str
    source_run_sha256: str | None = None
    suggestions: list[FollowupSuggestion] = Field(default_factory=list, max_length=4)
    unavailable_actions: tuple[str, ...] = ("multi_database_unavailable",)
