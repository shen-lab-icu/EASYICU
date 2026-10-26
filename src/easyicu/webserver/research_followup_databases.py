"""Optional host bridge to db6a decisions and the mdb1 dry compiler.

Neither owner exists in this checkout. Without an installed host bridge the
feature returns multi_database_unavailable, and no database launch is offered.
The bridge must list currently registered sources, call db6a for the reading,
and dry compile through mdb1, including any reviewed substitutions. A browser
or model cannot provide this bridge or its decisions.
"""
from __future__ import annotations

from typing import Protocol, Sequence

from easyicu.databases import normalize_database_key
from easyicu.research_agent.planning.capability_registry import resolve_primary_capability
from easyicu.research_agent.schema import AnalysisPlan, ResearchContext
from .research_followup_compile import require_database_family
from .research_followup_contracts import CompletedResult, DatabaseDecision, FollowupReading, OtherDatabaseDelta


class OtherDatabaseOwner(Protocol):
    def registered_databases(self) -> Sequence[str]: ...

    def decide(self, source: CompletedResult, *, database: str) -> DatabaseDecision: ...

    def dry_compile(self, source: CompletedResult, *, decision: DatabaseDecision) -> tuple[ResearchContext, AnalysisPlan]: ...


def other_database_deltas(source: CompletedResult, owner: OtherDatabaseOwner):
    reading = FollowupReading.from_context(source.context, family=source.plan.analysis_type)
    current = normalize_database_key(reading.database)
    databases = sorted({normalize_database_key(key) for key in owner.registered_databases()})
    for database in databases:
        if database == current:
            continue
        try:
            decision = owner.decide(source, database=database)
            if (decision.status not in {"exact", "substitution"} or decision.database != database
                    or decision.source_context_sha256 != reading.source_context_sha256):
                continue
            context, plan = owner.dry_compile(source, decision=decision)
            if normalize_database_key(context.cohort.database) != database or plan.analysis_type != source.plan.analysis_type:
                continue
            require_database_family(context, plan.analysis_type)
            verdict = resolve_primary_capability(analysis_type=plan.analysis_type, plan=plan)
            if not verdict.coherent or verdict.execution_owner != "host_deterministic" or verdict.scientific_validation == "unsupported":
                continue
            yield OtherDatabaseDelta(source_reading=reading,
                                     target_reading=FollowupReading.from_context(context, family=plan.analysis_type),
                                     capability_decision=decision, plan=plan)
        except (ValueError, KeyError):
            continue
